from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TrustItem:
    name: str
    label_ru: str


TRUST_COLUMNS: Final[tuple[TrustItem, ...]] = (
    TrustItem("trust_police", "доверие_полиция"),
    TrustItem("trust_courts", "доверие_суды"),
    TrustItem("trust_army", "доверие_армия"),
    TrustItem("trust_church", "доверие_церковь"),
    TrustItem("trust_media", "доверие_сми"),
    TrustItem("trust_government", "доверие_правительство"),
    TrustItem("trust_parliament", "доверие_парламент"),
    TrustItem("trust_parties", "доверие_партии"),
    TrustItem("trust_president", "доверие_президент"),
    TrustItem("trust_business", "доверие_крупный_бизнес"),
)


@dataclass(frozen=True)
class SocdemField:
    name: str
    kind: str  # "numeric" | "categorical"


SOCDEM_COLUMNS: Final[tuple[SocdemField, ...]] = (
    SocdemField("age", "numeric"),
    SocdemField("gender", "categorical"),
    SocdemField("education", "categorical"),
    SocdemField("income_subj", "categorical"),
    SocdemField("settlement", "categorical"),
    SocdemField("region_id", "categorical"),
)


GENDER_LABELS: Final[tuple[str, ...]] = ("мужской", "женский", "другое / отказ")

EDUCATION_LABELS: Final[tuple[str, ...]] = (
    "неполное среднее",
    "среднее",
    "среднее специальное",
    "неоконченное высшее",
    "высшее",
    "учёная степень",
)

INCOME_SUBJ_LABELS: Final[tuple[str, ...]] = (
    "хватает с трудом",
    "хватает",
    "скорее хватает",
    "скорее не хватает",
    "не хватает",
    "затрудняюсь ответить",
)

SETTLEMENT_LABELS: Final[tuple[str, ...]] = (
    "столица региона",
    "город 100–500 тыс.",
    "город 50–100 тыс.",
    "малый город / пгт",
    "село",
)

REGION_LABELS: Final[tuple[str, ...]] = tuple(f"REG_{i:02d}" for i in range(1, 21))


DEFAULT_N_ROWS: Final[int] = 5000
DEFAULT_N_COLS: Final[int] = 174
DEFAULT_SEED: Final[int] = 42

ATTITUDE_5: Final[tuple[str, ...]] = (
    "полностью не согласен(на)",
    "скорее не согласен(на)",
    "затрудняюсь ответить",
    "скорее согласен(на)",
    "полностью согласен(на)",
)

IMPORTANCE_5: Final[tuple[str, ...]] = (
    "совсем не важно",
    "не очень важно",
    "средне",
    "важно",
    "очень важно",
)

INTEREST_4: Final[tuple[str, ...]] = (
    "совсем не интересует",
    "мало интересует",
    "интересует",
    "очень интересует",
)

YES_NO_UNSURE: Final[tuple[str, ...]] = ("да", "нет", "затрудняюсь ответить")

FREQUENCY_5: Final[tuple[str, ...]] = (
    "никогда",
    "редко",
    "иногда",
    "часто",
    "постоянно",
)

CAT_POOLS_WIDE: Final[tuple[tuple[str, ...], ...]] = (
    ATTITUDE_5,
    IMPORTANCE_5,
    INTEREST_4,
    YES_NO_UNSURE,
    FREQUENCY_5,
)
