"""Generate synthetic gambling/betting ad texts with CTR/CR metrics.

Realistic ad copy patterns based on common industry approaches.
CTR/CR correlated with text features (urgency, numbers, emotion, etc.)
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# --- Ad copy templates by vertical ---

GAMBLING_TEMPLATES = [
    # High CTR patterns (urgency + numbers + excitement)
    "🎰 {bonus} бонус на первый депозит! Только {days} дней. Играй и выигрывай!",
    "Получи {bonus} фриспинов прямо сейчас! Регистрация за 30 секунд 🔥",
    "{count}+ игроков уже выиграли сегодня. Твоя очередь! Бонус {bonus}",
    "⚡ ГОРЯЩЕЕ ПРЕДЛОЖЕНИЕ: {bonus} бонус + {spins} фриспинов. Осталось {hours}ч",
    "Выиграй до {jackpot}€ с бонусом {bonus}! Начни с {min_dep}€ депозита",
    # Medium CTR patterns
    "Лучшие слоты онлайн. Бонус {bonus} на первый депозит. Играй сейчас",
    "Казино с лицензией. {count}+ игр. Бонус до {bonus} для новых игроков",
    "Попробуй удачу в {count}+ играх. Быстрые выплаты, бонус {bonus}",
    "Играй в лучшие слоты. Бонус {bonus}, вейджер x{wager}. Регистрируйся",
    "Надёжное казино с быстрыми выплатами. До {bonus} бонус на депозит",
    # Low CTR patterns (generic, no urgency)
    "Онлайн казино. Широкий выбор игр. Бонусы для игроков",
    "Играйте в казино онлайн. Множество слотов и настольных игр",
    "Казино с бонусами. Регистрация бесплатна. Начните играть",
    "Лучшее онлайн казино для вас. Попробуйте свою удачу сегодня",
    "Добро пожаловать в мир азарта. Играйте и выигрывайте призы",
]

BETTING_TEMPLATES = [
    # High CTR
    "⚽ Фрибет {bonus}€ без условий! Ставь на {league} сегодня. Коэфы до {odds}x",
    "LIVE ставки на {league}! Бонус {bonus}€ + кэшбэк {cashback}%. Начни сейчас 🔥",
    "{count}+ событий каждый день. Фрибет {bonus}€ при регистрации. Успей забрать!",
    "Ставки на спорт с коэфами до {odds}x. {bonus}€ фрибет — только {days} дней!",
    "🏆 {count} тысяч игроков выбрали нас. Фрибет {bonus}€. Присоединяйся!",
    # Medium CTR
    "Ставки на {league} и другие лиги. Бонус {bonus}€ для новых клиентов",
    "Букмекер с лицензией. {count}+ видов спорта. Фрибет до {bonus}€",
    "Делай ставки на спорт. Высокие коэффициенты, быстрые выплаты",
    "Спортивные ставки онлайн. Бонус {bonus}€, вывод за 24 часа",
    "Лучшие коэффициенты на {league}. Начни с бонуса {bonus}€",
    # Low CTR
    "Букмекерская контора. Ставки на спорт онлайн. Регистрация",
    "Делайте ставки на спортивные события. Широкая линия",
    "Спортивные ставки для всех. Простая регистрация. Начните сегодня",
    "Букмекер онлайн. Множество событий для ставок каждый день",
    "Ставки на спорт. Бонусы и акции для клиентов",
]

NUTRA_TEMPLATES = [
    # High CTR
    "Минус {kg}кг за {days} дней! {count}+ отзывов. Скидка {discount}% только сегодня 🔥",
    "Врачи в шоке: этот метод убирает {kg}кг без диет. Успей со скидкой {discount}%!",
    "⚡ ПОСЛЕДНИЙ ДЕНЬ: {product} со скидкой {discount}%. Уже {count}+ заказов",
    "{celebrity} раскрыл секрет: минус {kg}кг за {days} дней. Попробуй бесплатно!",
    "Похудей на {kg}кг без спорта и диет! {count}+ довольных клиентов. Закажи сейчас",
    # Medium CTR
    "Эффективное средство для похудения. Скидка {discount}% на первый заказ",
    "Натуральный продукт для снижения веса. До минус {kg}кг за месяц",
    "{product} — проверенное средство. {count}+ положительных отзывов",
    "Похудение без диет. {product} со скидкой {discount}%. Бесплатная доставка",
    "Снижение веса естественным путём. Закажи {product} сегодня",
    # Low CTR
    "Средство для похудения. Натуральные компоненты. Закажите",
    "Худейте эффективно с нашим продуктом. Доставка по всей стране",
    "Продукт для снижения веса. Попробуйте и оцените результат",
    "Натуральное средство для стройности. Простой заказ онлайн",
    "Начните путь к стройности сегодня. Наш продукт поможет вам",
]

GEOS = ['US', 'DE', 'GB', 'BR', 'IN', 'TR', 'ID', 'PH', 'MX', 'IT']
VERTICALS = ['gambling', 'betting', 'nutra']
TEMPLATES = {
    'gambling': GAMBLING_TEMPLATES,
    'betting': BETTING_TEMPLATES,
    'nutra': NUTRA_TEMPLATES,
}

FILL_VALUES = {
    'bonus': ['100%', '150%', '200%', '500€', '100€', '50€', '200€'],
    'days': ['3', '5', '7', '2'],
    'count': ['10,000', '50,000', '100,000', '5,000', '1,000'],
    'spins': ['50', '100', '200', '150'],
    'hours': ['6', '12', '24', '48'],
    'jackpot': ['10,000', '50,000', '100,000', '1,000,000'],
    'min_dep': ['5', '10', '20'],
    'wager': ['30', '35', '40', '45'],
    'league': ['Champions League', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A'],
    'odds': ['5', '10', '20', '50'],
    'cashback': ['10', '15', '20'],
    'kg': ['5', '8', '10', '12', '15'],
    'discount': ['30', '40', '50', '60', '70'],
    'product': ['SlimFit', 'KetoMax', 'BurnPro', 'NaturThin'],
    'celebrity': ['Известный блогер', 'Звезда ТВ', 'Фитнес-тренер'],
}


def fill_template(template: str) -> str:
    """Fill template placeholders with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = '{' + key + '}'
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def compute_text_features(text: str) -> dict:
    """Compute ground-truth features for CTR simulation."""
    has_number = any(c.isdigit() for c in text)
    has_urgency = any(w in text.lower() for w in [
        'только', 'осталось', 'сейчас', 'сегодня', 'последний',
        'горящ', 'успей', 'быстр',
    ])
    has_emoji = any(ord(c) > 0x2600 for c in text)
    has_social_proof = any(w in text.lower() for w in [
        'игроков', 'клиентов', 'отзывов', 'заказов', 'выбрали', 'выиграли',
    ])
    length = len(text)
    exclamation_count = text.count('!')

    return {
        'has_number': has_number,
        'has_urgency': has_urgency,
        'has_emoji': has_emoji,
        'has_social_proof': has_social_proof,
        'length': length,
        'exclamation_count': exclamation_count,
    }


def simulate_ctr(features: dict, vertical: str, geo: str, template_idx: int,
                 n_templates: int) -> float:
    """Simulate CTR based on text features with noise."""
    # Base CTR by vertical
    base = {'gambling': 0.035, 'betting': 0.030, 'nutra': 0.040}[vertical]

    # Template position effect (first = high CTR, last = low CTR)
    position_effect = 1.0 - 0.6 * (template_idx / (n_templates - 1))

    # Feature multipliers
    mult = 1.0
    if features['has_number']:
        mult *= 1.25
    if features['has_urgency']:
        mult *= 1.35
    if features['has_emoji']:
        mult *= 1.15
    if features['has_social_proof']:
        mult *= 1.20
    if features['exclamation_count'] >= 2:
        mult *= 1.10

    # Geo effect
    geo_mult = {
        'US': 1.3, 'DE': 1.2, 'GB': 1.25, 'BR': 0.8, 'IN': 0.6,
        'TR': 0.7, 'ID': 0.65, 'PH': 0.6, 'MX': 0.75, 'IT': 1.0,
    }.get(geo, 1.0)

    ctr = base * position_effect * mult * geo_mult
    ctr *= np.random.lognormal(0, 0.25)  # noise
    return round(min(ctr, 0.20), 5)


def simulate_cr(ctr: float, vertical: str) -> float:
    """Simulate CR from CTR (correlated but noisy)."""
    cr_ratio = {'gambling': 0.08, 'betting': 0.10, 'nutra': 0.06}[vertical]
    cr = ctr * cr_ratio * np.random.lognormal(0, 0.3)
    return round(min(cr, 0.05), 5)


def generate_dataset(n: int = 500) -> pd.DataFrame:
    rows = []
    for i in range(n):
        vertical = random.choice(VERTICALS)
        geo = random.choice(GEOS)
        templates = TEMPLATES[vertical]
        template_idx = random.randint(0, len(templates) - 1)
        template = templates[template_idx]
        ad_text = fill_template(template)

        features = compute_text_features(ad_text)
        ctr = simulate_ctr(features, vertical, geo, template_idx, len(templates))
        cr = simulate_cr(ctr, vertical)
        impressions = random.randint(5000, 200000)
        clicks = int(impressions * ctr)
        conversions = int(clicks * cr)

        rows.append({
            'ad_id': f'ad_{i:04d}',
            'ad_text': ad_text,
            'vertical': vertical,
            'geo': geo,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'ctr': ctr,
            'cr': cr,
        })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = generate_dataset(500)
    df.to_csv('ads_dataset.csv', index=False)
    print(f"Generated {len(df)} ads -> ads_dataset.csv")
    print(f"Verticals: {df['vertical'].value_counts().to_dict()}")
    print(f"CTR range: {df['ctr'].min():.4f} - {df['ctr'].max():.4f}")
    print(f"CR range: {df['cr'].min():.5f} - {df['cr'].max():.5f}")
    print(f"\nSample:")
    print(df[['ad_text', 'vertical', 'geo', 'ctr', 'cr']].head(5).to_string())
