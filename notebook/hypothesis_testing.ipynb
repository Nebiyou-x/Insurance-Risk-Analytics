
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ----------------------
# Config
# ----------------------
DATA_PATH = "data/insurance.csv"
REPORT_DIR = "reports"
TOP_N_ZIPCODES = 30          # number of zipcodes to include in zipcode-wide tests
MIN_POLICIES_PER_GROUP = 30  # ignore groups with fewer policies than this threshold
ALPHA = 0.05                 # significance level
np.random.seed(42)

# Ensure report dir exists
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------------
# Load & Basic Prep
# ----------------------
print("Loading data:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["TransactionMonth"], dayfirst=True, infer_datetime_format=True)

# standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# required columns check
req_cols = ["policyid", "transactionmonth", "postalcode", "province", "gender", "totalpremium", "totalclaims"]
missing_req = [c for c in req_cols if c not in df.columns]
if missing_req:
    raise ValueError(f"Missing required columns: {missing_req}")

# Create KPIs
# Claim occurrence (binary): 1 if totalclaims > 0 else 0
df["claim_flag"] = (df["totalclaims"].fillna(0) > 0).astype(int)

# Claim frequency per group will be proportion of claim_flag == 1
# Claim severity: average claim amount given claim occurred (exclude zero-claim policies)
df["claim_amount"] = df["totalclaims"].fillna(0)
df["claim_severity"] = df.loc[df["claim_amount"] > 0, "claim_amount"]  # sparse series

# Margin
df["margin"] = df["totalpremium"].fillna(0) - df["totalclaims"].fillna(0)

# Quick global metrics
total_policies = df["policyid"].nunique()
global_frequency = df["claim_flag"].mean()
global_severity = df.loc[df["claim_amount"] > 0, "claim_amount"].mean() if (df["claim_amount"] > 0).any() else 0
global_loss_ratio = df["totalclaims"].sum() / df["totalpremium"].sum()

print(f"Total policies: {total_policies:,}")
print(f"Global claim frequency: {global_frequency:.4f}")
print(f"Global avg claim severity (given claim): {global_severity:.2f}")
print(f"Global loss ratio: {global_loss_ratio:.3f}")

# ----------------------
# Helper functions
# ----------------------
def proportion_test(group_a_success, group_a_n, group_b_success, group_b_n):
    """
    Two-proportion z-test using normal approx.
    Returns z-stat, p-value.
    """
    p1 = group_a_success / group_a_n
    p2 = group_b_success / group_b_n
    p_pool = (group_a_success + group_b_success) / (group_a_n + group_b_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/group_a_n + 1/group_b_n))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def cohen_d(x, y):
    """Cohen's d for effect size between two samples"""
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.nanvar(x, ddof=1) + (ny-1)*np.nanvar(y, ddof=1)) / dof)
    if pooled_sd == 0:
        return 0.0
    return (np.nanmean(x) - np.nanmean(y)) / pooled_sd

def standardized_mean_diff(a, b):
    """SMD for numeric arrays (used for balance checks)"""
    return cohen_d(a, b)

# ----------------------
# Test 1: Provinces (Risk differences)
# For both frequency and severity
# ----------------------
print("\n\n=== TEST 1: Provinces (Claim Frequency & Claim Severity) ===")

province_table = df.groupby("province").agg(
    policies=("policyid", "nunique"),
    claims=("claim_flag", "sum"),
    freq=("claim_flag", "mean"),
    total_claim_amount=("claim_amount", "sum"),
    severity=lambda x: df.loc[x.index, "claim_amount"].replace(0, np.nan).dropna().mean()
).sort_values("policies", ascending=False)

province_table["severity"] = province_table["severity"].astype(float)
province_table = province_table.reset_index()

# Drop provinces with very few policies
province_table_filtered = province_table[province_table["policies"] >= MIN_POLICIES_PER_GROUP].copy()
print("Provinces included (policies >= {}): {}".format(MIN_POLICIES_PER_GROUP, province_table_filtered["province"].tolist()))

# Frequency test: Chi-square across provinces or overall ANOVA? We'll do chi-square contingency for counts.
# Build contingency: rows=province, columns=[claims, no_claims]
contingency = []
provinces = province_table_filtered["province"].tolist()
for p in provinces:
    row = province_table_filtered.loc[province_table_filtered["province"] == p].iloc[0]
    claims = int(row["claims"])
    no_claims = int(row["policies"] - row["claims"])
    contingency.append([claims, no_claims])

chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\nProvince claim-frequency chi-square: chi2={chi2:.3f}, p={p_chi:.6f}, dof={dof}")

# For severity (continuous but heavy-tailed): use Kruskal-Wallis across provinces (non-parametric)
# assemble severity samples per province (only claimed policies)
severity_samples = []
province_names_for_sev = []
for p in provinces:
    sub = df[(df["province"] == p) & (df["claim_amount"] > 0)]
    if len(sub) >= 5:  # need some samples
        severity_samples.append(sub["claim_amount"].values)
        province_names_for_sev.append(p)

if len(severity_samples) >= 2:
    kw_stat, p_kw = stats.kruskal(*severity_samples)
    print(f"Province claim-severity Kruskal-Wallis: H={kw_stat:.3f}, p={p_kw:.6f}")
else:
    print("Not enough claimed observations per province for Kruskal-Wallis severity test.")

# Save province table
province_table.to_csv(os.path.join(REPORT_DIR, "results_province_summary.csv"), index=False)


# ----------------------
# Test 2: Zipcodes (Risk differences between zip codes)
# Because many zipcodes exist we limit to top-N by premium
# ----------------------
print("\n\n=== TEST 2: Zipcodes (Claim Frequency & Severity) ===")
zip_prem = df.groupby("postalcode").agg(
    policies=("policyid", "nunique"), total_premium=("totalpremium", "sum"), claims=("claim_flag","sum")
).reset_index().dropna(subset=["postalcode"])

top_zips = zip_prem.sort_values("total_premium", ascending=False).head(TOP_N_ZIPCODES)
print("Top zipcodes by premium (count):", len(top_zips))

# Prepare tests: overall chi-sq across these zipcodes (frequency)
contingency_zip = []
zipcodes_in_test = []
for z in top_zips["postalcode"].tolist():
    sub = df[df["postalcode"] == z]
    if len(sub) >= MIN_POLICIES_PER_GROUP:
        claims = int(sub["claim_flag"].sum())
        no_claims = int(len(sub) - claims)
        contingency_zip.append([claims, no_claims])
        zipcodes_in_test.append(z)
    else:
        print(f"Skipping postalcode {z} due to small N = {len(sub)}")

if len(contingency_zip) >= 2:
    chi2_zip, p_zip_chi, dof_zip, expected_zip = stats.chi2_contingency(contingency_zip)
    print(f"Top-{len(contingency_zip)} zipcodes claim-frequency chi2={chi2_zip:.3f}, p={p_zip_chi:.6f}")
else:
    print("Not enough zipcodes with sufficient policies for a valid chi-square test.")

# Severity across zipcodes: Kruskal-Wallis using claimed amounts per zipcode
severity_samples_zip = []
zip_for_sev = []
for z in zipcodes_in_test:
    sub_claims = df[(df["postalcode"]==z) & (df["claim_amount"] > 0)]["claim_amount"]
    if len(sub_claims) >= 5:
        severity_samples_zip.append(sub_claims.values)
        zip_for_sev.append(z)

if len(severity_samples_zip) >= 2:
    kw_zip, p_kw_zip = stats.kruskal(*severity_samples_zip)
    print(f"Top zipcodes claim-severity Kruskal-Wallis: H={kw_zip:.3f}, p={p_kw_zip:.6f}")
else:
    print("Not enough claimed observations per zipcode for Kruskal-Wallis severity test.")

# For pairwise zip comparisons (if overall test significant), do pairwise two-proportion tests and adjust p-values
pairwise_results = []
if len(zipcodes_in_test) >= 2 and len(contingency_zip) >= 2:
    # compute pairwise z-tests for frequency
    for i in range(len(zipcodes_in_test)):
        for j in range(i+1, len(zipcodes_in_test)):
            zi = zipcodes_in_test[i]
            zj = zipcodes_in_test[j]
            subi = df[df["postalcode"]==zi]
            subj = df[df["postalcode"]==zj]
            ni = len(subi); nj = len(subj)
            si = int(subi["claim_flag"].sum()); sj = int(subj["claim_flag"].sum())
            if ni < MIN_POLICIES_PER_GROUP or nj < MIN_POLICIES_PER_GROUP:
                continue
            z_stat, pval = proportion_test(si, ni, sj, nj)
            pairwise_results.append({"zip_i":zi, "zip_j":zj, "n_i":ni, "n_j":nj, "s_i":si, "s_j":sj, "z":z_stat, "p":pval})

    # multiple testing correction on p-values
    if pairwise_results:
        pvals = [r["p"] for r in pairwise_results]
        reject, pvals_corr, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        for idx,r in enumerate(pairwise_results):
            r["p_adj"] = pvals_corr[idx]
            r["reject_null"] = bool(reject[idx])
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(os.path.join(REPORT_DIR, "results_zipcode_pairwise_freq.csv"), index=False)
    print("Saved pairwise zipcode frequency results (adjusted for multiple testing).")

# ----------------------
# Test 3: Margin difference between zip codes
# Use Kruskal-Wallis (non-parametric) across top zipcodes
# ----------------------
print("\n\n=== TEST 3: Margin differences across zipcodes ===")
margin_samples = []
zip_for_margin = []
for z in zipcodes_in_test:
    sub = df[df["postalcode"]==z]
    if len(sub) >= MIN_POLICIES_PER_GROUP:
        margin_samples.append(sub["margin"].values)
        zip_for_margin.append(z)

if len(margin_samples) >= 2:
    kw_margin, p_margin = stats.kruskal(*margin_samples)
    print(f"Margin Kruskal-Wallis across top zipcodes: H={kw_margin:.3f}, p={p_margin:.6f}")
else:
    print("Not enough data for margin Kruskal-Wallis test.")

# Save margin summary per zipcode
zip_margin_summary = df[df["postalcode"].isin(zipcodes_in_test)].groupby("postalcode").agg(
    policies=("policyid","nunique"),
    avg_margin=("margin","mean"),
    median_margin=("margin","median"),
    margin_std=("margin","std")
).reset_index()
zip_margin_summary.to_csv(os.path.join(REPORT_DIR, "results_zipcode_margin_summary.csv"), index=False)


# ----------------------
# Test 4: Gender differences (Claim Frequency & Severity)
# ----------------------
print("\n\n=== TEST 4: Gender differences (Claim Frequency & Severity) ===")
gender_table = df.groupby("gender").agg(
    policies=("policyid","nunique"),
    claims=("claim_flag","sum"),
    freq=("claim_flag","mean")
).reset_index()

if set(gender_table["gender"].dropna()) >= set(["Male","Female"]):
    male = df[df["gender"]=="Male"]
    female = df[df["gender"]=="Female"]
    nm = len(male); nf = len(female)
    sm = int(male["claim_flag"].sum()); sf = int(female["claim_flag"].sum())

    z_gender, p_gender = proportion_test(sm, nm, sf, nf)
    print(f"Gender frequency z={z_gender:.3f}, p={p_gender:.6f} (Male vs Female)")

    # severity: compare distributions using Mann-Whitney U (non-parametric)
    male_sev = male[male["claim_amount"] > 0]["claim_amount"]
    female_sev = female[female["claim_amount"] > 0]["claim_amount"]
    if len(male_sev) >= 5 and len(female_sev) >= 5:
        u_stat, p_u = stats.mannwhitneyu(male_sev, female_sev, alternative="two-sided")
        d = cohen_d(male_sev, female_sev)
        print(f"Gender severity Mann-Whitney U={u_stat:.3f}, p={p_u:.6f}, Cohen's d={d:.3f}")
    else:
        print("Insufficient claimed observations by gender for Mann-Whitney severity test.")
else:
    print("Gender categories Male/Female not both available; skipping gender tests.")

gender_table.to_csv(os.path.join(REPORT_DIR, "results_gender_summary.csv"), index=False)


# ----------------------
# Reporting decisions: Accept/Reject Nulls + Business Interpretation
# ----------------------
print("\n\n=== DECISIONS & BUSINESS INTERPRETATION ===")
decisions = []

# Provinces: base on chi-square and severity KW
province_reject_freq = (p_chi < ALPHA)
province_reject_sev = ('p_kw' in locals() and p_kw < ALPHA)

decisions.append({
    "hypothesis":"No risk differences across provinces (frequency)",
    "test":"Chi-square (counts)",
    "p_value":p_chi,
    "reject": province_reject_freq
})
decisions.append({
    "hypothesis":"No risk differences across provinces (severity)",
    "test":"Kruskal-Wallis (severity)",
    "p_value": p_kw if 'p_kw' in locals() else np.nan,
    "reject": province_reject_sev
})

# Zipcodes: use overall chi-square & KW results
zip_reject_freq = ('p_zip_chi' in locals() and p_zip_chi < ALPHA)
zip_reject_sev = ('p_kw_zip' in locals() and p_kw_zip < ALPHA)

decisions.append({
    "hypothesis":"No risk differences between zip codes (frequency, top N)",
    "test":"Chi-square (counts)",
    "p_value": p_zip_chi if 'p_zip_chi' in locals() else np.nan,
    "reject": zip_reject_freq
})
decisions.append({
    "hypothesis":"No risk differences between zip codes (severity, top N)",
    "test":"Kruskal-Wallis (severity)",
    "p_value": p_kw_zip if 'p_kw_zip' in locals() else np.nan,
    "reject": zip_reject_sev
})

# Zipcode margin
zip_margin_reject = ('p_margin' in locals() and p_margin < ALPHA)
decisions.append({
    "hypothesis":"No margin difference between zip codes (top N)",
    "test":"Kruskal-Wallis (margin)",
    "p_value": p_margin if 'p_margin' in locals() else np.nan,
    "reject": zip_margin_reject
})

# Gender
gender_reject_freq = (p_gender < ALPHA) if 'p_gender' in locals() else False
gender_reject_sev = ('p_u' in locals() and p_u < ALPHA)

decisions.append({
    "hypothesis":"No risk difference between Women and Men (frequency)",
    "test":"Two-proportion z-test",
    "p_value": p_gender if 'p_gender' in locals() else np.nan,
    "reject": gender_reject_freq
})
decisions.append({
    "hypothesis":"No risk difference between Women and Men (severity)",
    "test":"Mann-Whitney U",
    "p_value": p_u if 'p_u' in locals() else np.nan,
    "reject": gender_reject_sev
})

dec_df = pd.DataFrame(decisions)
dec_df.to_csv(os.path.join(REPORT_DIR, "results_hypothesis_decisions.csv"), index=False)
print(dec_df.to_string(index=False))

# Print interpretative recommendations
print("\nINTERPRETATIONS & BUSINESS RECOMMENDATIONS (TEMPLATE):")
for r in decisions:
    hyp = r["hypothesis"]
    pval = r["p_value"]
    rej = r["reject"]
    if np.isnan(pval):
        status = "INDETERMINATE (insufficient data)"
    else:
        status = "REJECT" if rej else "FAIL TO REJECT"

    print(f"\n- {hyp}")
    print(f"  Test: {r['test']}  p = {pval:.6g}. Decision: {status}.")
    if rej:
        print("  Business interpretation (example):")
        print("    • Statistically significant difference detected. Consider segment-specific pricing.")
        print("    • Example action: For provinces/zipcodes with higher frequency or severity, consider raising premiums or tightening underwriting; for low-risk zipcodes, run A/B pricing experiments offering lower premiums to capture market share.")
    elif not np.isnan(pval):
        print("  Business interpretation (example):")
        print("    • No strong evidence of difference at alpha = {:.3f}. Consider pooling or further stratified analysis (e.g., by vehicle type).".format(ALPHA))
    else:
        print("  Business interpretation: Data insufficient; collect more samples or aggregate groups.")


print("\nAll result CSVs saved to:", REPORT_DIR)
print("Task 3 completed (script-level). Use these CSVs and printed output for the interim submission.")
