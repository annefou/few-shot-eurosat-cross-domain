# Snakefile — Few-Shot EuroSAT Cross-Domain Replication
#
# Usage:
#   snakemake --cores 1       # run experiment
#   snakemake --cores 1 -n    # dry run

RESULTS = "results"

rule all:
    input:
        f"{RESULTS}/cross_domain_results.json",
        f"{RESULTS}/cross_domain_eurosat.png",

rule run_experiment:
    output:
        f"{RESULTS}/cross_domain_results.json",
        f"{RESULTS}/cross_domain_eurosat.png",
    log:
        f"{RESULTS}/logs/01_cross_domain_eurosat.log",
    shell:
        """
        mkdir -p {RESULTS}/logs
        jupytext --to notebook --execute 01_cross_domain_eurosat.py 2>&1 | tee {log}
        """
