# DVSA FOI Data — Summary of Findings & Future Work

**Source:** Jake Cracknell's FOI request to DVSA (2023 data)  
**Data:** 1M+ anonymised test records, 328 centres, 55 fault categories  
**Repo:** github.com/JakeCracknell/dvsa_driving_test_data  
**Files used here:** Annex B (test-level), Annex C (fault-level), Annex D (aggregated faults),  
`dtc_summary.csv` (per-centre summary), pivot tables generated from Annex D.

---

## 1. What Jake Found

### Pass rates vary enormously — but not randomly

National mean pass rate is **52%**, ranging from 13.8% to 100% across 328 centres.  
In Yorkshire specifically: **38.1% (Grimsby Coldwater) to 61.2% (Bridlington)** — a 23 percentage  
point spread across centres that are geographically close. This is not explained by candidate  
ability alone.

### Time of day doesn't matter much

Pass rates by hour range from 46% (06:00) to 51% (17:00). Jake concluded there is no  
meaningful advantage to booking at a particular time of day. Useful to cite as a confounder  
we can dismiss.

### Attempt number barely matters either

| Attempt | Pass rate |
|---|---|
| 1st | 48.8% |
| 2nd | 49.6% |
| 3rd | 49.0% |
| 4th | 48.2% |
| 5th+ | 46.7% |

Pass rates are remarkably flat across attempt numbers. Repeat takers are not significantly  
more likely to pass — suggesting the test difficulty (route complexity, road environment) is  
a persistent barrier, not candidate preparation.

### Fault categories predict pass rate

Nationally, the fault categories most negatively correlated with pass rate (across all 328 centres):

| Fault | Correlation with pass rate |
|---|---|
| Response to traffic signs | −0.39 |
| Judgement — crossing | −0.29 |
| Controlled stop — Promptness | −0.29 |
| Response to road markings | −0.29 |
| Response to other road users | −0.27 |

In **Yorkshire specifically**, the signal is stronger and junctions dominate:

| Fault | Correlation with pass rate |
|---|---|
| Junctions — turning right | −0.63 |
| Response to traffic lights | −0.58 |
| Positioning — normal driving | −0.55 |
| Signals — correctly | −0.53 |
| Mirrors — change speed | −0.53 |
| Maintain progress — undue hesitation | −0.51 |
| Junctions — turning left | −0.48 |
| Junctions — observation | −0.46 |

Junction faults and traffic light response are the dominant predictors of poor pass rates  
in Yorkshire. This is consistent with urban complexity being the primary environmental driver.

### Some Yorkshire centres are structural outliers

Relative fault rates vs national average (>1.0 = worse than national):

| Centre | Junctions right | Traffic lights | Hesitation | Use of speed |
|---|---|---|---|---|
| Doncaster | 1.66× | 1.25× | 1.38× | 1.79× |
| Hull | 1.54× | 0.77 | 0.96 | 0.67 |
| Sheffield (Handsworth) | 1.27× | 0.96 | 1.43× | 1.51× |
| Leeds | 1.26× | 1.49× | 1.10× | 0.89 |

Doncaster stands out — worse than national average on junction turns, hesitation, *and*  
speed. This is a mixed signature (complex junctions AND fast roads) which is unusual.  
Hull has high junction turning faults but low traffic light and speed faults — consistent  
with a layout that has complex turns but less signal-controlled infrastructure.

### First-attempt pass rates reveal strategic repeat booking

Negative gap (first attempt *lower* than overall) means repeat bookers from other areas  
are inflating the overall rate:

| Centre | First attempt | Overall | Gap |
|---|---|---|---|
| Grimsby Coldwater | 35.1% | 38.1% | −3.0% |
| Halifax | 41.0% | 43.2% | −2.2% |
| Pontefract | 42.8% | 45.0% | −2.1% |
| Northallerton | 47.7% | 49.9% | −2.2% |

These centres appear harder than they are on first-attempt rates — candidates are travelling  
there to find availability, not because it's easy. First-attempt pass rate is the cleaner  
measure of genuine structural difficulty.

---

## 2. What Jake's Analysis Implies — Future Work

### The geographic question is unanswered

Jake identified *that* centres differ and *which fault categories* drive the differences.  
He did not explain *why* — i.e. what road network characteristics around each centre produce  
those fault patterns. That is the gap our pipeline fills.

**Hypothesis:** Centres with high junction-turning fault rates will have high `degree_mean`  
and `betweenness_relative` in their catchment. Centres with high hesitation/speed faults  
will have longer link lengths, higher speed limits, and rural road classifications.  
Centres with high traffic light faults will have higher `pop_density` and more signalised  
junctions in the catchment.

This is testable with `dtc_catchment_model.py` using our existing risk scores and network  
features. If the correlations hold, it is a publishable finding.

### Per-fault rather than per-pass modelling

Rather than predicting overall pass rate from road features, model each fault category  
rate separately. This produces five testable sub-hypotheses with cleaner causal stories  
and is more useful to DVSA — it tells them not just which centres are hard but *what kind*  
of road exposure is causing specific fault patterns.

### First-attempt pass rate as the target variable

Overall pass rates are inflated by strategic repeat bookers at some centres. Models should  
use first-attempt rates as the target, not overall. We have computed this from Annex B for  
all Yorkshire centres — it is ready to use.

### Fault co-occurrence structure

The correlation between fault categories across centres has not been explored. A cluster  
analysis of centres by their fault profile would identify whether there are distinct  
"hard centre types" (e.g. complex urban vs fast rural vs mixed) that correspond to  
identifiable road network signatures. This is a natural extension once the basic  
catchment model is running.

### The route data gap

The most significant limitation is that fault location within the route is unknown. The  
FOI data records faults per test but not where on the route they occurred. With GPS route  
data (from plotaroute or similar), fault rates could be linked to specific road links rather  
than catchment averages. This would move the analysis from "is this centre's environment  
hard" to "is this specific junction causing failures". That is a fundamentally more  
actionable finding for DVSA/local highway authorities.

Jake noted that DVSA holds a text field recording where faults occurred. A follow-up FOI  
requesting that field — even in aggregate form (e.g. "junction X on road Y, N serious faults  
in 2023") — would unlock this analysis without requiring individual test record linkage.

### Panel data via follow-up FOI

The dataset covers 2023 only. Annual variation in pass rates (e.g. due to road changes,  
new junctions, traffic signal installations) cannot be distinguished from structural effects.  
A follow-up FOI for 2021 and 2022 data in the same format would allow a fixed-effects model  
that controls for centre-level baseline differences and identifies whether changes in road  
infrastructure correlate with changes in fault rates over time. This is the strongest  
possible causal design available without randomisation.

---

## 3. Data Available

| File | Contents | Status |
|---|---|---|
| `dtc_summary.csv` | Pass rate, manoeuvre rates, lat/lon per centre | ✅ Have |
| `agg_2023_fails_pivot.csv` | Normalised S+D fault rates per centre × fault type | ✅ Generated |
| `agg_2023_fails_relative_pivot.csv` | Same, as ratio vs national average | ✅ Generated |
| `agg_2023_minors_pivot.csv` | Minor fault rates per centre × fault type | ✅ Generated |
| First-attempt pass rates | Computed from Annex B | ✅ Computed |
| Annex B | Per-test: TC, date, time, attempt, result | ✅ Have (1M rows) |
| Annex C | Per-test: all 55 fault categories (raw) | ✅ Have (truncated at 1M — Excel limit) |
| Annex D | Aggregated fault counts by centre × manoeuvre | ✅ Have |
| GPS test routes | Crowdsourced GPX from plotaroute.com | 🔄 Collecting |
| 2021/2022 FOI data | Same format, prior years | ❌ Not yet requested |
| DVSA fault location field | Where on route each fault occurred | ❌ Not publicly available |