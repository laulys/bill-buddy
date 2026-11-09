"""Who owes whom: trip balances and the transfers that settle them.

Everything works in whole cents (integers), so balances always add up to zero.

The settle-up uses the classic greedy approach: repeatedly let the person who
is owed the most get paid by the person who owes the most. It needs at most
(people - 1) transfers. Finding the absolute minimum number of transfers is
NP-hard in general, so greedy is the standard practical choice (it's what
apps like Splitwise do in spirit).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

Transfer = Tuple[str, str, int]  # (from, to, cents)


def net_balances(members: Iterable[str], expenses: Iterable[Tuple[str, int, Mapping[str, int]]]) -> Dict[str, int]:
    """Net cents per person: positive = others owe them, negative = they owe.

    Each expense is (payer, total_cents, {person: share_cents}).
    """
    net = {name: 0 for name in members}
    for payer, total, shares in expenses:
        net[payer] += total
        for person, share in shares.items():
            net[person] -= share
    return net


def settle_up(net: Mapping[str, int]) -> List[Transfer]:
    """Transfers (from, to, cents) that bring every balance to zero."""
    if sum(net.values()) != 0:
        raise ValueError("Balances must add up to zero.")
    creditors = {name: cents for name, cents in net.items() if cents > 0}
    debtors = {name: -cents for name, cents in net.items() if cents < 0}
    transfers: List[Transfer] = []
    while creditors and debtors:
        # Biggest creditor and biggest debtor; ties broken by name, so results are stable.
        to = max(creditors, key=lambda n: (creditors[n], n))
        frm = max(debtors, key=lambda n: (debtors[n], n))
        cents = min(creditors[to], debtors[frm])
        transfers.append((frm, to, cents))
        for side, name in ((creditors, to), (debtors, frm)):
            side[name] -= cents
            if side[name] == 0:
                del side[name]
    return transfers
