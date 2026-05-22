"""Benchmark problems live here.

Phase 1: empty marker. Problems P1-P5 land in Phase 2.

When a problem module is added, importing it must register a subclass of
``bench.problems.base.BenchProblem`` so that ``run_bench.py`` can discover
it via ``BenchProblem.__subclasses__()``.
"""
