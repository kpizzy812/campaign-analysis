"""Load test for /predict endpoint.

Target: 120 RPS x 60 seconds = 7200 requests
Criteria: P95 < 100ms, RPS >= 115, Error rate < 1%

Uses a pool of ~200 campaigns to simulate realistic cache hit patterns.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import aiohttp

# Pool of realistic campaign features for load test
GEOS = ['US', 'DE', 'GB', 'BR', 'IN', 'TR', 'ID', 'PH', 'MX', 'IT']
VERTICALS = ['ecommerce', 'gambling', 'nutra', 'dating', 'finance', 'crypto', 'sweepstakes']
SOURCES = ['facebook', 'google', 'tiktok', 'push', 'native', 'inapp']
DEVICES = ['mobile', 'desktop', 'tablet']
OS_LIST = ['android', 'ios', 'windows', 'macos']
BUDGETS = [10, 20, 30, 50, 100, 200, 500]


def generate_campaign_pool(n: int = 200) -> list[dict]:
    """Generate a pool of campaigns that will be reused (realistic cache hits)."""
    random.seed(42)
    pool = []
    for _ in range(n):
        pool.append({
            "geo": random.choice(GEOS),
            "vertical": random.choice(VERTICALS),
            "traffic_source": random.choice(SOURCES),
            "device": random.choice(DEVICES),
            "os": random.choice(OS_LIST),
            "bid": round(random.uniform(0.05, 2.0), 3),
            "daily_budget": random.choice(BUDGETS),
            "hour": random.randint(0, 23),
            "dow": random.randint(0, 6),
        })
    return pool


@dataclass
class LoadTestResult:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    rps_achieved: float = 0.0
    latencies: list = field(default_factory=list)

    def summary(self) -> str:
        error_rate = self.failed / max(self.total_requests, 1) * 100
        lines = [
            "=" * 50,
            "  LOAD TEST RESULTS",
            "=" * 50,
            f"  Total requests:  {self.total_requests:,}",
            f"  Successful:      {self.successful:,}",
            f"  Failed:          {self.failed:,}",
            f"  Error rate:      {error_rate:.2f}%  (target: < 1%)",
            f"",
            f"  P50 latency:     {self.p50_latency_ms:.1f}ms",
            f"  P95 latency:     {self.p95_latency_ms:.1f}ms  (target: < 100ms)",
            f"  P99 latency:     {self.p99_latency_ms:.1f}ms",
            f"",
            f"  RPS achieved:    {self.rps_achieved:.1f}  (target: >= 115)",
            "=" * 50,
        ]

        # Verdict
        p95_ok = self.p95_latency_ms < 100
        rps_ok = self.rps_achieved >= 115
        err_ok = error_rate < 1

        lines.append(f"  P95 < 100ms:     {'PASS' if p95_ok else 'FAIL'}")
        lines.append(f"  RPS >= 115:      {'PASS' if rps_ok else 'FAIL'}")
        lines.append(f"  Error < 1%:      {'PASS' if err_ok else 'FAIL'}")

        return "\n".join(lines)


async def _send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    latencies: list,
    results: dict,
    sem: asyncio.Semaphore,
):
    """Send a single request and record latency."""
    async with sem:
        start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                await resp.read()
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
                if resp.status == 200:
                    results['ok'] += 1
                else:
                    results['fail'] += 1
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            results['fail'] += 1


async def load_test(
    api_url: str = "http://localhost:8000/predict",
    target_rps: int = 120,
    duration_seconds: int = 60,
) -> LoadTestResult:
    """Run load test at target RPS for given duration.

    Uses a pool of 200 campaigns — ~60% requests will hit same features (cache).
    """
    pool = generate_campaign_pool(200)
    total_requests = target_rps * duration_seconds
    interval = 1.0 / target_rps

    latencies = []
    results = {'ok': 0, 'fail': 0}
    sem = asyncio.Semaphore(target_rps * 2)  # allow burst

    connector = aiohttp.TCPConnector(limit=target_rps * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        test_start = time.perf_counter()

        for i in range(total_requests):
            payload = random.choice(pool)
            task = asyncio.create_task(
                _send_request(session, api_url, payload, latencies, results, sem)
            )
            tasks.append(task)

            # Pace requests to target RPS
            elapsed = time.perf_counter() - test_start
            expected_time = (i + 1) * interval
            if expected_time > elapsed:
                await asyncio.sleep(expected_time - elapsed)

        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - test_start

    # Calculate percentiles
    latencies.sort()
    n = len(latencies)

    result = LoadTestResult(
        total_requests=results['ok'] + results['fail'],
        successful=results['ok'],
        failed=results['fail'],
        p50_latency_ms=latencies[int(n * 0.50)] if n else 0,
        p95_latency_ms=latencies[int(n * 0.95)] if n else 0,
        p99_latency_ms=latencies[int(n * 0.99)] if n else 0,
        rps_achieved=n / total_time if total_time > 0 else 0,
        latencies=latencies,
    )

    return result


async def main():
    print("Starting load test: 120 RPS x 60s = 7,200 requests\n")
    print("Make sure the API server is running:")
    print("  uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4\n")

    result = await load_test("http://localhost:8000/predict", target_rps=120, duration_seconds=60)
    print(result.summary())

    # Also check /health for cache stats
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/health") as resp:
                health = await resp.json()
                print(f"\n  Cache stats: {json.dumps(health.get('cache', {}), indent=2)}")
        except Exception:
            print("\n  Could not fetch /health")

    return result


if __name__ == "__main__":
    asyncio.run(main())
