"""
🚀 api/main.py — Vercel Entrypoint
This file MUST be at /api/main.py (not savant_engine/api/main.py)

Vercel scans for:
  ✅ /api/main.py
  ✅ /api/index.py
  ❌ savant_engine/api/main.py (won't detect this)
"""

import os
import sys
import logging

# Add repo root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RRF_API')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    rrf_api_url = os.getenv('RRF_API_URL', 'https://antonypamo-apirrf.hf.space')
    hf_token = os.getenv('HF_TOKEN', '')
    supabase_url = os.getenv('SUPABASE_URL', '')
    supabase_key = os.getenv('SUPABASE_KEY', '')
    
    rate_limits = {'FREE': 10, 'PRO': 100, 'BUSINESS': 500, 'ENTERPRISE': None}

config = Config()

# ============================================================================
# SUPABASE (Optional)
# ============================================================================

try:
    from supabase import create_client
    supabase_client = None
    if config.supabase_key:
        try:
            supabase_client = create_client(config.supabase_url, config.supabase_key)
            logger.info('✅ Supabase connected')
        except Exception as e:
            logger.warning(f'⚠️  Supabase unavailable: {e}')
except ImportError:
    supabase_client = None
    logger.warning('⚠️  Supabase not installed')

# ============================================================================
# DATABASE LAYER
# ============================================================================

def save_api_call(api_key: str, query: str, num_docs: int, latency_ms: float, status: str):
    """Log API call to Supabase."""
    if not supabase_client:
        return
    try:
        supabase_client.table('api_calls').insert({
            'api_key_id': api_key,
            'query_text': query[:500],
            'num_documents': num_docs,
            'latency_ms': latency_ms,
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'created_at': datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.debug(f'Could not log to Supabase: {e}')

def get_api_key_quota(api_key: str) -> Dict[str, Any]:
    """Get API key quota from Supabase."""
    if not supabase_client:
        return {'tier': 'FREE', 'used': 0, 'limit': 1000, 'is_within_quota': True}
    
    try:
        key_resp = supabase_client.table('api_keys').select('*').eq('key_id', api_key).limit(1).execute()
        if key_resp.data:
            key_data = key_resp.data[0]
            tier = key_data.get('tier', 'FREE')
            
            today = datetime.utcnow()
            month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            calls_resp = supabase_client.table('api_calls').select('id', count='exact').gte(
                'timestamp', month_start.isoformat()
            ).eq('api_key_id', api_key).execute()
            
            used = calls_resp.count or 0
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

# ============================================================================
# RRF API CLIENT
# ============================================================================

class RRFClient:
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        return session
    
    def rerank(self, query: str, documents: List[str]) -> tuple:
        """Call RRF API to rerank documents. Returns: (results, latency_ms, status)"""
        start = time.time()
        
        try:
            resp = self.session.post(
                f'{config.rrf_api_url}/v1/rerank',
                json={'query': query, 'documents': documents, 'alpha': 0.2},
                headers={'Authorization': f'Bearer {config.hf_token}'},
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
            logger.warning(f'API timeout after 30s')
            return [], latency, 'timeout'
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f'RRF API error: {e}')
            return [], latency, 'error'

api_client = RRFClient()

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, api_key: str, tier: str) -> bool:
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
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title='RRF Savant API v5.4',
    description='Production rank fusion engine',
    version='5.4.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# ============================================================================
# MODELS
# ============================================================================

class RankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    documents: List[str] = Field(..., min_items=1, max_items=100)

class RankResult(BaseModel):
    id: int
    rank: int
    score: float

class RankResponse(BaseModel):
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
    """Health check."""
    try:
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
            'database': {'ok': supabase_client is not None}
        }
    }

@app.post('/api/rank', response_model=RankResponse)
async def rank_documents(
    request: RankRequest,
    authorization: str = Header(None),
    background_tasks: BackgroundTasks = None
) -> RankResponse:
    """Rank documents."""
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing API key')
    
    api_key = authorization.replace('Bearer ', '').strip()
    
    try:
        # Get quota
        quota = get_api_key_quota(api_key)
        tier = quota.get('tier', 'FREE')
        
        # Check rate limit
        if not limiter.is_allowed(api_key, tier):
            raise HTTPException(status_code=429, detail='Rate limit exceeded')
        
        # Check quota
        if not quota.get('is_within_quota', True):
            raise HTTPException(status_code=402, detail='Quota exceeded')
        
        # Call RRF API
        results, latency, status = api_client.rerank(request.query, request.documents)
        
        # Log to database (background task)
        if background_tasks:
            background_tasks.add_task(
                save_api_call,
                api_key, request.query, len(request.documents), latency, status
            )
        
        # Format results
        formatted_results = []
        if status == 'success':
            for result in results[:10]:
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
        raise HTTPException(status_code=500, detail='Internal server error')

@app.get('/api/usage')
async def get_usage(authorization: str = Header(None)) -> Dict[str, Any]:
    """Get API key usage."""
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing API key')
    
    api_key = authorization.replace('Bearer ', '').strip()
    quota = get_api_key_quota(api_key)
    
    today = datetime.utcnow()
    reset_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    return {
        'tier': quota.get('tier'),
        'used': quota.get('used'),
        'limit': quota.get('limit'),
        'remaining': quota.get('limit') - quota.get('used') if quota.get('limit') else None,
        'reset_date': reset_date.isoformat()
    }

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
            'usage': 'GET /api/usage'
        }
    }
