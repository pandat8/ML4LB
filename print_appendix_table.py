import pandas as pd
import re

def generate_latex_table(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Extract the columns relevant to the 3600s main results
    cols_to_keep = [
        'instance',
        'baseline_solve_time', 'baseline_final_gap',
        'freq0_solve_time', 'freq0_final_gap',
        'freq100_solve_time', 'freq100_final_gap'
    ]
    df = df[cols_to_keep]

    # Start constructing the LaTeX longtable
    latex = [
        r"% Add these packages to your preamble if not already there:",
        r"% \usepackage{longtable}",
        r"% \usepackage{booktabs}",
        r"% \usepackage{multirow}",
        r"% \usepackage{xcolor}  <-- Add this for color",
        r"",
        r"\begin{center}",
        r"\color{blue}",
        r"\footnotesize",
        r"\begin{longtable}{l r r r r r r}",
        r"\caption{Detailed per-instance results on the MIPLIB 2017 dataset (3600s time limit). Times are reported in seconds and gaps in percentage (averaged over 5 random seeds).} \label{tab:miplib_detailed} \\",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Instance}} & \multicolumn{2}{c}{\textbf{SCIP-baseline}} & \multicolumn{2}{c}{\textbf{lb-freq0}} & \multicolumn{2}{c}{\textbf{lb-freq100}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & \textbf{Time} & \textbf{Gap (\%)} & \textbf{Time} & \textbf{Gap (\%)} & \textbf{Time} & \textbf{Gap (\%)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"",
        r"\multicolumn{7}{c}%",
        r"{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Instance}} & \multicolumn{2}{c}{\textbf{SCIP-baseline}} & \multicolumn{2}{c}{\textbf{lb-freq0}} & \multicolumn{2}{c}{\textbf{lb-freq100}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & \textbf{Time} & \textbf{Gap (\%)} & \textbf{Time} & \textbf{Gap (\%)} & \textbf{Time} & \textbf{Gap (\%)} \\",
        r"\midrule",
        r"\endhead",
        r"",
        r"\midrule \multicolumn{7}{r}{{Continued on next page}} \\",
        r"\endfoot",
        r"",
        r"\bottomrule",
        r"\endlastfoot",
        r""
    ]

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Shorten the instance name by removing the prefix, e.g., "miplib2017_binary-41_transformed_p0201" -> "p0201"
        instance_full = str(row['instance'])
        instance_short = re.sub(r'miplib2017_binary-\d+_transformed_', '', instance_full)
        # LaTeX requires underscores to be escaped
        instance = instance_short.replace("_", "\\_")

        def format_val(val, is_time=False):
            if pd.isna(val):
                return "-"
            if is_time:
                # Format time to 1 decimal place
                return f"{val:.1f}"
            else:
                # Format gap to 2 decimal places
                return f"{val:.2f}"

        b_time = format_val(row['baseline_solve_time'], is_time=True)
        b_gap = format_val(row['baseline_final_gap'])

        f0_time = format_val(row['freq0_solve_time'], is_time=True)
        f0_gap = format_val(row['freq0_final_gap'])

        f100_time = format_val(row['freq100_solve_time'], is_time=True)
        f100_gap = format_val(row['freq100_final_gap'])

        latex.append(f"{instance} & {b_time} & {b_gap} & {f0_time} & {f0_gap} & {f100_time} & {f100_gap} \\\\")

    # Close the environments
    latex.append(r"\end{longtable}")
    latex.append(r"\end{center}")

    return "\n".join(latex)

if __name__ == "__main__":
    csv_filename = "./result/plots/scip_comparison_details_miplib2017_binary_-small_rootsol_seeds_averaged_v2_202607.csv"
    latex_code = generate_latex_table(csv_filename)
    
    # Print to console (or you could write to a file: open('appendix_table.tex', 'w').write(latex_code))
    print(latex_code)