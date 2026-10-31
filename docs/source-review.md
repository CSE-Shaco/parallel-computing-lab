# Original Coursework Review and Publication Gate

Reviewed private source: root parallel assignment 1 and 2 ZIP archives, their `.ipynb` code cells. No original notebook, report, supplied function, or embedded output is included in this candidate repository.

| Original finding | Independent implementation decision |
| --- | --- |
| Monte Carlo seeds used random-device and wall-clock values per thread/mode | Use indexed deterministic samples with one explicit seed |
| Unprotected shared counter intentionally invoked a data race | Explain the failure; provide only defined serial/local/mutex modes |
| Shared mutex acquired for each hit | Implement a small independent comparator with exactly defined locking scope |
| OpenMP notebook explicitly labels `explore_region` and difficulty initialization as supplied | Replace both with a new integer task and configurable uniform/tail layout |
| Worker durations measured after implicit OpenMP barrier | Record durations before final waiting and expose iteration totals |
| Auto shown repeatedly with chunk labels | One auto configuration per layout; require chunk=0 sentinel |
| Notebook evidence not controlled or independently revalidated | Store only new local raw measurements and independent tests |

The source files here are new portfolio code prepared with coding-agent assistance, not a claim that this refactor was part of the historical course work. General algorithmic ideas and assignment themes overlap; implementation was written independently rather than copied/renamed.

Publication classifications: new code/docs are **Requires Modification** until the owner reviews accuracy and academic constraints; original notebooks/reports remain **Keep Private**, and professor-provided code requires permission before any redistribution. No claim of academic-policy clearance is made. No LICENSE file is generated. Confirm course policy and ownership before any remote publication; local git initialization does not grant publication rights.
