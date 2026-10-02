"""Discrete-event simulation: a small event-scheduling engine and three models built on it.

``Simulator`` keeps a future-event list (a binary heap ordered by time, ties broken by insertion order) and a
clock; events are callbacks scheduled at absolute times. ``TimeWeighted`` accumulates the time average of a
piecewise-constant quantity (queue length, busy servers, inventory) and ``Tally`` the average of
observations (waits). The models are a multi-server queue with time-varying Poisson arrivals, general service
and abandonment (the call centre of the project), an (s, S) inventory system with random demand and lead
times, and a machine-repair (finite-source) system - each small enough to read in one sitting.

>>> import numpy as np
>>> from engstoch import des
>>> sim = des.Simulator()
>>> log = []
>>> sim.schedule(2.0, lambda: log.append(sim.now)); sim.schedule(1.0, lambda: log.append(sim.now))
>>> sim.run()
>>> log
[1.0, 2.0]
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque

import numpy as np


class Simulator:
    """Event-scheduling simulation clock with a future-event list."""

    def __init__(self):
        self.now = 0.0
        self._events: list = []
        self._seq = itertools.count()
        self.n_events = 0

    def schedule(self, time: float, action, *args) -> None:
        """Schedule action(*args) at absolute time `time` (>= now)."""
        if time < self.now - 1e-12:
            raise ValueError("cannot schedule an event in the past")
        heapq.heappush(self._events, (time, next(self._seq), action, args))

    def after(self, delay: float, action, *args) -> None:
        self.schedule(self.now + delay, action, *args)

    def run(self, until: float = math.inf) -> None:
        """Process events in time order until the list is empty or the next event is after `until`."""
        while self._events and self._events[0][0] <= until:
            time, _, action, args = heapq.heappop(self._events)
            self.now = time
            self.n_events += 1
            action(*args)
        if until < math.inf:
            self.now = until


class TimeWeighted:
    """Time-weighted statistic of a piecewise-constant value (e.g. the number in queue)."""

    def __init__(self, sim: Simulator, value: float = 0.0, start: float = 0.0):
        self.sim, self.value, self.last, self.area, self.start = sim, value, start, 0.0, start
        self.max = value

    def update(self, value: float) -> None:
        self.area += self.value * (self.sim.now - self.last)
        self.last, self.value = self.sim.now, value
        self.max = max(self.max, value)

    def mean(self) -> float:
        area = self.area + self.value * (self.sim.now - self.last)
        return area / (self.sim.now - self.start) if self.sim.now > self.start else 0.0

    def reset(self) -> None:
        """Discard the history (end of a warm-up period)."""
        self.area, self.last, self.start = 0.0, self.sim.now, self.sim.now


class Tally:
    """Running mean, variance (Welford) and count of observations."""

    def __init__(self):
        self.n, self._mean, self._m2, self.values = 0, 0.0, 0.0, []

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self._mean
        self._mean += d / self.n
        self._m2 += d * (x - self._mean)
        self.values.append(x)

    def mean(self) -> float:
        return self._mean if self.n else math.nan

    def var(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else math.nan


def call_centre(
    rate_fn,
    rate_max: float,
    service_sampler,
    agents,
    t_end: float,
    rng,
    patience_sampler=None,
    answer_within: float = 20 / 3600,
    warmup: float = 0.0,
) -> dict:
    """Multi-server FCFS queue with non-homogeneous Poisson arrivals (thinning of a rate_max process),
    service times from service_sampler(rng), optional abandonment after patience_sampler(rng), and a staffing
    level agents(t) (a number or a function of time; a reduction takes effect as agents finish calls).
    Returns per-call records and the time-averaged queue length and occupancy."""
    sim = Simulator()
    staff = agents if callable(agents) else (lambda t, a=agents: a)
    queue: deque = deque()
    busy = [0]
    q_len = TimeWeighted(sim)
    busy_tw = TimeWeighted(sim)
    calls = []

    def arrival():
        if rng.random() < rate_fn(sim.now) / rate_max:
            call = {"arrival": sim.now, "start": math.nan, "end": math.nan, "abandoned": False}
            calls.append(call)
            if busy[0] < staff(sim.now) and not queue:
                start_service(call)
            else:
                queue.append(call)
                q_len.update(len(queue))
                if patience_sampler is not None:
                    sim.after(patience_sampler(rng), abandon, call)
        nxt = sim.now + rng.exponential(1.0 / rate_max)
        if nxt <= t_end:
            sim.schedule(nxt, arrival)

    def start_service(call):
        call["start"] = sim.now
        busy[0] += 1
        busy_tw.update(busy[0])
        sim.after(service_sampler(rng), departure, call)

    def departure(call):
        call["end"] = sim.now
        busy[0] -= 1
        busy_tw.update(busy[0])
        while queue and busy[0] < staff(sim.now):
            nxt = queue.popleft()
            q_len.update(len(queue))
            start_service(nxt)

    def abandon(call):
        if math.isnan(call["start"]) and not call["abandoned"]:
            call["abandoned"] = True
            call["end"] = sim.now
            queue.remove(call)
            q_len.update(len(queue))

    def staffing_check():  # a staffing increase takes effect immediately
        while queue and busy[0] < staff(sim.now):
            nxt = queue.popleft()
            q_len.update(len(queue))
            start_service(nxt)
        if sim.now + 0.25 <= t_end:
            sim.after(0.25, staffing_check)

    sim.schedule(rng.exponential(1.0 / rate_max), arrival)
    if callable(agents):
        sim.schedule(0.0, staffing_check)
    if warmup > 0:
        sim.schedule(warmup, lambda: (q_len.reset(), busy_tw.reset()))
    sim.run()
    arr = np.array([c["arrival"] for c in calls])
    start = np.array([c["start"] for c in calls])
    ab = np.array([c["abandoned"] for c in calls], dtype=bool)
    keep = arr >= warmup
    wait = np.where(ab, np.nan, start - arr)
    answered = (~ab) & (wait <= answer_within)
    return {
        "arrival": arr[keep],
        "wait": wait[keep],
        "abandoned": ab[keep],
        "service_level": float(answered[keep].mean()) if keep.any() else math.nan,
        "abandon_rate": float(ab[keep].mean()) if keep.any() else math.nan,
        "mean_queue": q_len.mean(),
        "mean_busy": busy_tw.mean(),
        "n_events": sim.n_events,
    }


def inventory_sS(
    s: int,
    S: int,
    demand_rate: float,
    demand_size_sampler,
    lead_time_sampler,
    t_end: float,
    rng,
    holding: float = 1.0,
    shortage: float = 10.0,
    order_cost: float = 50.0,
    review: float | None = None,
) -> dict:
    """(s, S) inventory with compound-Poisson demand, backorders and random lead times. With continuous review
    (review=None) an order up to S is placed when the inventory position falls to s or below; with periodic
    review every `review` time units. Costs: holding per unit-time on hand, shortage per unit-time backordered,
    and a fixed cost per order. Returns the long-run cost rate and its parts."""
    sim = Simulator()
    level = [S]  # net inventory (negative = backorders)
    position = [S]  # net inventory plus outstanding orders
    on_hand = TimeWeighted(sim, S)
    backlog = TimeWeighted(sim, 0)
    orders = [0]

    def set_level(v):
        level[0] = v
        on_hand.update(max(v, 0))
        backlog.update(max(-v, 0))

    def maybe_order():
        if position[0] <= s:
            q = S - position[0]
            position[0] += q
            orders[0] += 1
            sim.after(lead_time_sampler(rng), receive, q)

    def receive(q):
        set_level(level[0] + q)

    def demand():
        d = int(demand_size_sampler(rng))
        set_level(level[0] - d)
        position[0] -= d
        if review is None:
            maybe_order()
        sim.after(rng.exponential(1.0 / demand_rate), demand)

    def periodic():
        maybe_order()
        sim.after(review, periodic)

    sim.after(rng.exponential(1.0 / demand_rate), demand)
    if review is not None:
        sim.schedule(0.0, periodic)
    sim.run(until=t_end)
    h, b = on_hand.mean(), backlog.mean()
    cost = holding * h + shortage * b + order_cost * orders[0] / t_end
    return {"cost_rate": cost, "mean_on_hand": h, "mean_backlog": b, "order_rate": orders[0] / t_end}


def machine_repair(n_machines: int, n_repairers: int, fail_rate: float, repair_sampler, t_end: float, rng) -> dict:
    """Finite-source queue: n machines each failing at rate fail_rate when up, repaired FCFS by n_repairers with
    general repair times. Returns the time-averaged number of machines down and the availability."""
    sim = Simulator()
    down = TimeWeighted(sim, 0)
    queue: deque = deque()
    busy = [0]
    n_down = [0]

    def fail():
        n_down[0] += 1
        down.update(n_down[0])
        if busy[0] < n_repairers:
            busy[0] += 1
            sim.after(repair_sampler(rng), repaired)
        else:
            queue.append(1)

    def repaired():
        n_down[0] -= 1
        down.update(n_down[0])
        sim.after(rng.exponential(1.0 / fail_rate), fail)
        if queue:
            queue.popleft()
            sim.after(repair_sampler(rng), repaired)
        else:
            busy[0] -= 1

    for _ in range(n_machines):
        sim.after(rng.exponential(1.0 / fail_rate), fail)
    sim.run(until=t_end)
    return {"mean_down": down.mean(), "availability": 1 - down.mean() / n_machines}
