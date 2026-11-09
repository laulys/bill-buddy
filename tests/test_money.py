"""Money math: exact cents, splits that always add up, clear errors."""

import pytest

from bill_buddy import money
from bill_buddy.money import MoneyError


def test_tip():
    assert money.calculate_tip(60, 20) == {"bill": 60.0, "tip_percent": 20.0, "tip": 12.0, "total_with_tip": 72.0}


def test_tip_rounds_half_up_to_cents():
    assert money.calculate_tip(10.05, 15)["tip"] == 1.51  # 1.5075 -> 1.51
    assert money.calculate_tip(0.1 + 0.2, 100)["total_with_tip"] == 0.6  # no float noise


def test_split_evenly_adds_up_exactly():
    result = money.split_evenly(100, 3)
    assert result["shares"] == [33.34, 33.33, 33.33]
    assert result["note"].startswith("1 person pays 1 cent more")
    assert round(sum(result["shares"]), 2) == 100


def test_split_evenly_without_leftover_has_no_note():
    assert money.split_evenly(253, 4) == {"total": 253.0, "people": 4, "shares": [63.25] * 4}


@pytest.mark.parametrize("total, people", [(0.05, 7), (999.99, 13), (1, 100), (12345.67, 9)])
def test_split_evenly_never_loses_a_cent(total, people):
    shares = money.split_evenly(total, people)["shares"]
    assert round(sum(shares), 2) == total
    assert max(shares) - min(shares) <= 0.01 + 1e-9


def test_split_itemized_example():
    result = money.split_itemized([("Ana", 18), ("Luis", 32)], shared_amount=24, tip_percent=15)
    assert result["per_person"]["Ana"] == {"own_items": 18.0, "shared_part": 12.0, "tax": 0.0, "tip": 4.5, "pays": 34.5}
    assert result["per_person"]["Luis"]["pays"] == 50.6
    assert result["grand_total"] == 85.1


def test_split_itemized_merges_names_and_adds_up_with_tax_and_tip():
    result = money.split_itemized(
        [("Ana", 18), ("Luis", 32), ("ana ", 4.5)], shared_amount=24, tip_percent=15, tax_percent=8
    )
    people = result["per_person"]
    assert list(people) == ["Ana", "Luis"]
    assert people["Ana"]["own_items"] == 22.5
    assert round(sum(p["pays"] for p in people.values()), 2) == result["grand_total"] == 96.56
    assert round(sum(p["tip"] for p in people.values()), 2) == result["tip"]
    assert round(sum(p["tax"] for p in people.values()), 2) == result["tax"]


def test_allocate_is_proportional_and_exact():
    from decimal import Decimal
    parts = money.allocate(1000, [Decimal(1), Decimal(2), Decimal(3)])
    assert parts == [167, 333, 500]
    assert sum(parts) == 1000


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: money.calculate_tip(-5, 10), "positive"),
        (lambda: money.calculate_tip(50, 150), "too high"),
        (lambda: money.calculate_tip("abc", 10), "must be a number"),
        (lambda: money.split_evenly(10, 0), "whole number between 1 and 100"),
        (lambda: money.split_evenly(10, 2.5), "whole number"),
        (lambda: money.split_itemized([]), "at least one item"),
        (lambda: money.split_itemized([(" ", 5)]), "person's name"),
    ],
)
def test_bad_input_gives_a_clear_error(call, message):
    with pytest.raises(MoneyError, match=message):
        call()
