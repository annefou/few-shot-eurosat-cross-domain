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
        "figures/replication_comparison.png",

rule plot_replication:
    """Headline comparison figure - seconds, no GPU, reads only committed results."""
    input:
        f"{RESULTS}/cross_domain_results.json",
    output:
        "figures/replication_comparison.png",
    shell:
        "python 04_plot_replication.py"


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
