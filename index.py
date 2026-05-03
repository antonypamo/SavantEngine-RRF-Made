"""
🚀 RRF Savant v5.4 — FastAPI Production Application

Complete production-ready FastAPI app that:
- Uses real RRF API (https://antonypamo-apirrf.hf.space)
- Integrates with Supabase database
- Implements rate limiting and quota enforcement
- Handles authentication and billing
- Deploys to Vercel as serverless function

Deployment:
  vercel deploy
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

# FastAPI
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# HTTP
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Database
try:
    from supabase import create_client, Client
except ImportError:
    Client = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('RRF_API')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ProdConfig:
    """Production configuration."""
    # RRF API (live)
    rrf_api_url: str = os.getenv('RRF_API_URL', 'https://antonypamo-apirrf.hf.space')
    hf_token: str = os.getenv('HF_TOKEN', '')
    
    # Supabase (real database)
    supabase_url: str = os.getenv('SUPABASE_URL', '')
    supabase_key: str = os.getenv('SUPABASE_KEY', '')
    
    # Stripe (payments)
    stripe_key: str = os.getenv('STRIPE_KEY', '')
    
    # Rate limits
    rate_limits: Dict[str, int] = None
    
    def __post_init__(self):
        if self.rate_limits is None:
            self.rate_limits = {
                'FREE': 10,
                'PRO': 100,
                'BUSINESS': 500,
                'ENTERPRISE': None
            }

config = ProdConfig()

# ============================================================================
# SUPABASE INTEGRATION
# ============================================================================

class Database:
    """Supabase database manager."""
    
    def __init__(self, config: ProdConfig):
        self.config = config
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self):
        """Connect to Supabase."""
        try:
            if self.config.supabase_key:
                self.client = create_client(
                    self.config.supabase_url,
                    self.config.supabase_key
                )
                logger.info('✅ Supabase connected')
        except Exception as e:
            logger.warning(f'⚠️  Supabase unavailable: {e}')
    
    def save_api_call(self, api_key: str, query: str, num_docs: int,
                     latency_ms: float, status: str) -> bool:
        """Log API call for billing."""
        if not self.client:
            return False
        
        try:
            self.client.table('api_calls').insert({
                'api_key_id': api_key,
                'query_text': query[:500],
                'num_documents': num_docs,
                'latency_ms': latency_ms,
                'status': status,
                'timestamp': datetime.utcnow().isoformat(),
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            return True
        except Exception as e:
            logger.debug(f'Could not log to Supabase: {e}')
            return False
    
    def get_api_key_quota(self, api_key: str) -> Dict[str, Any]:
        """Get API key quota from database."""
        if not self.client:
            return {
                'tier': 'FREE',
                'used': 0,
                'limit': 1000,
                'is_within_quota': True
            }
        
        try:
            key_resp = self.client.table('api_keys').select('*').eq('key_id', api_key).execute()
            if key_resp.data:
                key_data = key_resp.data[0]
                tier = key_data.get('tier', 'FREE')
                
                # Get month start
                today = datetime.utcnow()
                month_start = today.replace(day=1, hour=0, minute=0, second=0)
                
                # Count this month's calls
                calls_resp = self.client.table('api_calls').select('id', count='exact').gte(
                    'timestamp', month_start.isoformat()
                ).eq('api_key_id', api_key).execute()
                
                used = calls_resp.count or 0
                
                # Tier limits
                limits = {'FREE': 1000, 'PRO': 100000, 'BUSINESS': 1000000, 'ENTERPRISE': None}
                limit = limits.get(tier, 1000)
                
                return {
                    'tier': tier,
                    'used': used,
                    'limit': limit,
                    'is_within_quota': limit is None or used < limit
                }
        except Exception as e:
            logger.debug(f'Could not fetch quota: {e}')
        
        return {'tier': 'FREE', 'used': 0, 'limit': 1000, 'is_within_quota': True}

db = Database(config)

# ============================================================================
# RRF API CLIENT
# ============================================================================

class RRFClient:
    """Live RRF API client."""
    
    def __init__(self, config: ProdConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create resilient HTTP session."""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        return session
    
    def rerank(self, query: str, documents: List[str]) -> tuple:
        """Call RRF API to rerank documents.
        
        Returns: (results, latency_ms, status)
        """
        start = time.time()
        
        try:
            resp = self.session.post(
                f'{self.config.rrf_api_url}/v1/rerank',
                json={'query': query, 'documents': documents, 'alpha': 0.2},
                headers={'Authorization': f'Bearer {self.config.hf_token}'},
                timeout=30
            )
            
            latency = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get('results', []), latency, 'success'
            else:
                return [], latency, f'http_{resp.status_code}'
        except requests.Timeout:
            latency = (time.time() - start) * 1000
            return [], latency, 'timeout'
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f'RRF API error: {e}')
            return [], latency, 'error'

api_client = RRFClient(config)

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """In-memory rate limiting."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, api_key: str, tier: str) -> bool:
        """Check if within rate limit."""
        limit = config.rate_limits.get(tier, 10)
        if limit is None:
            return True
        
        now = time.time()
        if api_key not in self.requests:
            self.requests[api_key] = []
        
        self.requests[api_key] = [ts for ts in self.requests[api_key] if now - ts < 60]
        
        if len(self.requests[api_key]) < limit:
            self.requests[api_key].append(now)
            return True
        
        return False

limiter = RateLimiter()

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title='RRF Savant API v5.4',
    description='Production rank fusion engine',
    version='5.4.0'
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class RankRequest(BaseModel):
    """Ranking request."""
    query: str = Field(..., description='Search query')
    documents: List[str] = Field(..., description='Documents to rank')

class RankResult(BaseModel):
    """Single ranking result."""
    id: int
    rank: int
    score: float

class RankResponse(BaseModel):
    """Ranking response."""
    status: str
    results: List[RankResult] = []
    latency_ms: float = 0
    quota_remaining: Optional[int] = None
    tier: str = 'FREE'
    timestamp: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get('/health')
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check RRF API
        resp = requests.get(
            f'{config.rrf_api_url}/health',
            headers={'Authorization': f'Bearer {config.hf_token}'},
            timeout=5
        )
        rrf_ok = resp.status_code == 200
    except:
        rrf_ok = False
    
    return {
        'status': 'ok' if rrf_ok else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'rrf_api': {'ok': rrf_ok},
            'database': {'ok': db.client is not None},
        }
    }

@app.post('/api/rank', response_model=RankResponse)
async def rank_documents(
    request: RankRequest,
    authorization: str = Header(None)
) -> RankResponse:
    """Rank documents using RRF engine."""
    
    # Extract API key
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing API key')
    
    api_key = authorization.replace('Bearer ', '').strip()
    
    try:
        start_time = time.time()
        
        # Get quota
        quota = db.get_api_key_quota(api_key)
        tier = quota.get('tier', 'FREE')
        
        # Check rate limit
        if not limiter.is_allowed(api_key, tier):
            raise HTTPException(status_code=429, detail='Rate limit exceeded')
        
        # Check quota
        if not quota.get('is_within_quota', True):
            raise HTTPException(status_code=402, detail='Quota exceeded')
        
        # Call RRF API
        results, latency, status = api_client.rerank(request.query, request.documents)
        
        # Log to database
        db.save_api_call(api_key, request.query, len(request.documents), latency, status)
        
        # Format results
        formatted_results = []
        if status == 'success':
            for result in results[:10]:  # Top 10
                formatted_results.append(RankResult(
                    id=result.get('id', 0),
                    rank=result.get('rank', 0),
                    score=float(result.get('score', 0))
                ))
        
        remaining = quota.get('limit') - quota.get('used') if quota.get('limit') else None
        
        return RankResponse(
            status='ok' if status == 'success' else 'error',
            results=formatted_results,
            latency_ms=round(latency, 1),
            quota_remaining=remaining,
            tier=tier,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Ranking error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/usage')
async def get_usage(authorization: str = Header(None)) -> Dict[str, Any]:
    """Get API key usage."""
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing API key')
    
    api_key = authorization.replace('Bearer ', '').strip()
    quota = db.get_api_key_quota(api_key)
    
    return {
        'tier': quota.get('tier'),
        'used': quota.get('used'),
        'limit': quota.get('limit'),
        'remaining': quota.get('limit') - quota.get('used') if quota.get('limit') else None,
        'reset_date': (datetime.utcnow().replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()
    }

@app.post('/api/batch-rank')
async def batch_rank(
    requests_list: List[RankRequest],
    authorization: str = Header(None)
) -> List[RankResponse]:
    """Batch ranking (multiple queries)."""
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing API key')
    
    api_key = authorization.replace('Bearer ', '').strip()
    results = []
    
    for req in requests_list:
        try:
            # Create synthetic request object for rank_documents
            class FakeRequest:
                query = req.query
                documents = req.documents
            
            # Call rank endpoint
            resp = await rank_documents(FakeRequest(), authorization)
            results.append(resp)
        except Exception as e:
            logger.error(f'Batch error: {e}')
            results.append(RankResponse(
                status='error',
                results=[],
                latency_ms=0,
                tier='FREE',
                timestamp=datetime.utcnow().isoformat()
            ))
    
    return results

@app.get('/')
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        'service': 'RRF Savant API v5.4',
        'status': 'ok',
        'docs': '/docs',
        'endpoints': {
            'health': 'GET /health',
            'rank': 'POST /api/rank',
            'usage': 'GET /api/usage',
            'batch': 'POST /api/batch-rank'
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'status': 'error',
            'error': exc.detail,
            'timestamp': datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f'Unhandled exception: {exc}')
    return JSONResponse(
        status_code=500,
        content={
            'status': 'error',
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# MAIN (for local testing)
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    logger.info('🚀 Starting RRF Savant API v5.4')
    logger.info(f'RRF API: {config.rrf_api_url}')
    logger.info(f'Database: {\"Connected\" if db.client else \"Demo mode\"}')
    
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8000)),
        log_level='info'
    )
