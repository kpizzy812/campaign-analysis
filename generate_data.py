"""
Генератор синтетического датасета рекламных кампаний.
Запускается ОТДЕЛЬНО от ноутбука анализа.

Выходной файл: campaigns_synthetic.csv
Схема: campaign_id, geo, vertical, traffic_source, device, os,
       bid, daily_budget, impressions, clicks, conversions, spend, revenue, created_at

Паттерны заложены реалистичные (на основе открытых бенчмарков digital advertising),
но конкретные числа — результат моделирования, не реальные данные.
"""

import pandas as pd
import numpy as np

np.random.seed(2024)
N = 8000

# --- Справочники ---
geos = ['US', 'DE', 'GB', 'BR', 'IN', 'ID', 'PH', 'TH', 'MX', 'TR',
        'FR', 'JP', 'IT', 'NG', 'PL']
geo_weights = [0.18, 0.09, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05,
               0.04, 0.04, 0.04, 0.03, 0.08]

verticals = ['ecommerce', 'gambling', 'nutra', 'dating', 'finance', 'crypto', 'sweepstakes']
vert_weights = [0.20, 0.18, 0.15, 0.14, 0.13, 0.12, 0.08]

sources = ['facebook', 'google', 'tiktok', 'push', 'native', 'inapp']
source_weights = [0.25, 0.22, 0.15, 0.15, 0.13, 0.10]

devices = ['mobile', 'desktop', 'tablet']
device_weights = [0.65, 0.28, 0.07]

os_map = {
    'mobile': ['android', 'ios'],
    'desktop': ['windows', 'macos'],
    'tablet': ['android', 'ios'],
}

budgets = [10, 20, 30, 50, 100, 200, 500]
budget_weights = [0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05]

# --- Генерация базовых полей ---
geo = np.random.choice(geos, N, p=geo_weights)
vertical = np.random.choice(verticals, N, p=vert_weights)
traffic_source = np.random.choice(sources, N, p=source_weights)
device = np.random.choice(devices, N, p=device_weights)
os_col = [np.random.choice(os_map[d]) for d in device]
daily_budget = np.random.choice(budgets, N, p=budget_weights)
bid = np.round(np.random.lognormal(mean=-1.5, sigma=0.7, size=N).clip(0.01, 3.0), 3)

# --- Временные метки с реалистичной структурой ---
# 6 месяцев, с суточными и недельными паттернами
start = pd.Timestamp('2024-01-01')
end = pd.Timestamp('2024-06-30')
date_range = pd.date_range(start, end, freq='h')
# Больше кампаний в будни, меньше в выходные
hour_weights = np.array([0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.06,
                         0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06,
                         0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02])
dow_weights = np.array([1.2, 1.2, 1.1, 1.1, 1.0, 0.7, 0.7])

ts_weights = np.array([
    hour_weights[ts.hour] * dow_weights[ts.dayofweek]
    for ts in date_range
])
# Добавляем тренд — больше кампаний к концу периода
days_from_start = np.array([(ts - start).days for ts in date_range])
trend = 1.0 + 0.3 * (days_from_start / days_from_start.max())
ts_weights = ts_weights * trend
ts_weights = ts_weights / ts_weights.sum()

created_at = np.random.choice(date_range, N, p=ts_weights)

# --- Impressions (зависят от бюджета и bid) ---
base_impr = daily_budget / bid * 1000
noise = np.random.lognormal(0, 0.5, N)
impressions = np.maximum(500, (base_impr * noise)).astype(int)

# --- CTR зависит от источника и устройства ---
source_ctr = {'facebook': 0.035, 'google': 0.045, 'tiktok': 0.025,
              'push': 0.055, 'native': 0.030, 'inapp': 0.040}
device_ctr_mult = {'mobile': 1.0, 'desktop': 1.15, 'tablet': 0.85}

base_ctr = np.array([source_ctr[s] for s in traffic_source])
dev_mult = np.array([device_ctr_mult[d] for d in device])
# Часовой модификатор
hour_ctr_mult = {h: 0.8 + 0.4 * np.sin(np.pi * (h - 6) / 12) for h in range(24)}
created_at_pd = pd.DatetimeIndex(created_at)
hour_mult = np.array([hour_ctr_mult[h] for h in created_at_pd.hour])

ctr = base_ctr * dev_mult * hour_mult * np.random.lognormal(0, 0.3, N)
ctr = ctr.clip(0.005, 0.15)
clicks = np.maximum(0, np.random.binomial(impressions, ctr))

# --- Conversion rate зависит от гео и вертикали ---
geo_cr = {'US': 0.06, 'DE': 0.055, 'GB': 0.05, 'FR': 0.045, 'JP': 0.05,
          'IT': 0.04, 'BR': 0.03, 'IN': 0.02, 'ID': 0.025, 'PH': 0.025,
          'TH': 0.03, 'MX': 0.028, 'TR': 0.03, 'NG': 0.015, 'PL': 0.04}
vert_cr_mult = {'ecommerce': 1.3, 'gambling': 0.7, 'nutra': 1.1, 'dating': 0.9,
                'finance': 1.0, 'crypto': 0.6, 'sweepstakes': 0.8}

cr = np.array([geo_cr[g] * vert_cr_mult[v] for g, v in zip(geo, vertical)])
cr = cr * np.random.lognormal(0, 0.4, N)
cr = cr.clip(0.001, 0.20)
conversions = np.maximum(0, np.random.binomial(clicks, cr))

# --- Spend = clicks * actual_cpc (cpc ~ bid * competition) ---
competition = np.random.uniform(0.7, 1.3, N)
actual_cpc = bid * competition
spend = np.round(clicks * actual_cpc, 2)

# --- Revenue зависит от вертикали и гео ---
vert_rev = {'ecommerce': 15, 'gambling': 45, 'nutra': 25, 'dating': 12,
            'finance': 35, 'crypto': 50, 'sweepstakes': 8}
geo_rev_mult = {'US': 1.5, 'DE': 1.3, 'GB': 1.4, 'FR': 1.2, 'JP': 1.6,
                'IT': 1.0, 'BR': 0.6, 'IN': 0.3, 'ID': 0.4, 'PH': 0.35,
                'TH': 0.5, 'MX': 0.5, 'TR': 0.45, 'NG': 0.25, 'PL': 0.7}

rev_per_conv = np.array([
    vert_rev[v] * geo_rev_mult[g] * np.random.lognormal(0, 0.5)
    for v, g in zip(vertical, geo)
])
revenue = np.round(conversions * rev_per_conv, 2)

# --- Сборка DataFrame ---
campaign_ids = [f'camp_{i:05d}' for i in range(N)]

df = pd.DataFrame({
    'campaign_id': campaign_ids,
    'geo': geo,
    'vertical': vertical,
    'traffic_source': traffic_source,
    'device': device,
    'os': os_col,
    'bid': bid,
    'daily_budget': daily_budget,
    'impressions': impressions,
    'clicks': clicks,
    'conversions': conversions,
    'spend': np.round(spend, 2),
    'revenue': np.round(revenue, 2),
    'created_at': created_at,
})

df = df.sort_values('created_at').reset_index(drop=True)
df.to_csv('campaigns_synthetic.csv', index=False)

print(f"Generated {len(df)} campaigns -> campaigns_synthetic.csv")
print(f"Date range: {df['created_at'].min()} -> {df['created_at'].max()}")
print(f"ROI > 0: {(df['revenue'] > df['spend']).mean():.1%}")
print(f"Columns: {list(df.columns)}")
