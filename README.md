# growth_rates

This repository is for quickly determining microbial growth rates in high-throughput from plate reader OD600 data. It uses the iPython (Jupyter Notebooks) environment to import raw plate reader data from an Excel export, find a best-fit linear regression to Time (x) v. ln(OD600) (y) data, quality check fits, and export fit parameters back to Excel for easy end-user data manipulation and visualization. To start, make sure you have installed Jupyter Notebooks or JupyterLab. I recommend using [Anaconda Navigator](https://www.anaconda.com/download) to download and access these tools.


## Scientific background
Bacteria and other microorganisms grown in liquid culture is typically modeled in four sequential phases: 
1. **lag phase**, where microbes are acclimated to being introduced to a new environment and therefore do not grow quickly,
2. **exponential phase**, where the microbial population undergoes rapid expansion limiting only by the organism's growth rate,
3. **stationary phase**, where the culture is now so dense that microbes must compete for limited resources and slow their growth to increase long-term chances of survival, and
4. **death phase**, where nutrients are depleted to the point that the culture experiences a large-scale die-off.

Of these four phases, *we are typically interested in modeling exponential phase*, as this can provide insight into the organism's fitness under given culture conditions with no other competing factors.

Shown in the image below you can see the first three phases of the growth of bacteria in liquid culture: exponential phase has been modeled by a black curve superimposed on the blue raw data values, with lag phase preceding and stationary phase following exponential phase. growth_rates generated the black model curve as the best overall fit to the blue data curve.

![image](https://github.com/user-attachments/assets/f7947a17-3b5c-4a5b-89e1-7dd6dc9a4f9f)


## Statistical regression analysis and window picking

In order to find the best model of exponential phase, the "raw" OD600 data is first log-transformed, generating a linear relationship between time and ln(OD600) for the exponential phase that can be modeled using the LinearRegression tool in scikit-learn. However, this relationship is only linear for a short time during exponential phase, so this program applies a sliding window algorithm to identify the best fit during exponential phase. This algorithm attempts linear regressions across the entire range of data at various window "sizes", then selects the best fit by maximizing the coefficient of determination (R^2) of the fitted line. To best model data, several further constraints are included with exact values based on trial-and-error, but these values may need to be adjusted with differing culture conditions:
1. OD600 values are forced to increase at least ~4.5-fold over the time window to filter out non-exponential phase fits
2. OD600 value at the end of the fitted time window must be at least ~0.05 to filter out fits from noise at low absorbances
3. Best-fit R^2 value must be >0.99 to classify as a valid fit to filter out fitting to subpar data
4. Minimum window size must include at least 100 minutes of culture time and 8 measurements

The code determining all of these constraints can be found in the "well_fitter" function and its "find_linear_regime" sub-function.

## Sampling, variables, and naming
This pipeline was developed for utility of bench-facing scientists in mind: in order to designate different samples, blank wells, and wells to drop from analysis, they simply need to designate variable attributes in a second sheet on their data file with column labels designated in row 2 (for reference, check the associated example_data). Based on these conventions, the program automatically identifies replicates, background subtracts samples using the blank wells, and IDs samples according to their well and collected variable attributes. 

**Required column labels** (order unimportant):
* "Well": These should be within the set of alphanumerics created from the product of (A-P)(1-24), e.g., "A24", "F6", "P15". The allowed set includes all wells on a standard 96- or 384-well plate.
* "Sample": This column is essential because it is where blank and dropped wells should be designated. **Wells labeled "blank" will be used for background subtraction;** the program requires at least one blank well per media type per experiment to run. By default, the program will also attempt to check all blank wells for contamination via a statistical method compared to other blank well timepoints and/or crossing an OD600 threshold of 0.5. Contamination-checking behavior can be turned off within the data_cleaner function. **Wells labeled "drop" are designated as manually excluded from analysis and ignored.** Otherwise, variable attributes in this column are treated as the same as variable attributes in other columns for the purpose of grouping.
* "Media": This column is essential because it is used to group blank wells during background subtraction. Setting all values in this column to the same value is allowed and results in background subtraction being performed once on the entire plate.

As many other non-essential column labels and attributes as desired can be designated, with analysis automatically using variable attributes from all columns except "Well" to identify sample replicates. If multiple plates/experiments are being analyzed at once, it is recommended to ensure that column labels are consistent across all plates.

**DO NOT DESIGNATE** column labels as any of the following, as these labels are generated and used internally by the program:\
 ["Assay", "Temperature", "Value", "Experiment"]


## Path and directory (folder) setup
Make sure this script is saved in the same folder as a folder labeled 'xlsx' containing your plate reader data files saved with the .xlsx extension.

Additionally, please modify the file_target variable to a string that is contained in the data files you want to analyze. Multiple data files can be analyzed at once if they all contain this same string. Note that these files will each be assigned a unique label by file name under the 'Experiment' column in the resulting dataframe.

For example, using file_target 'expt' will find and analyze '/expt1.xlsx', '/expt2.xlsx', and '/expt3.xlsx' in the 'xlsx' folder together, then tag each of the resulting subdatasets as 'expt1', 'expt2', and 'expt3' respectively.


## To-do
### Possible future updates:
* Providing additional import functionality for a range of Excel raw data formats
* Adding API support for plugging into larger experimental workflows that use growth parameters for model training
* Adding support for fitting growth data to the Gompertz curve
* More metadata functionality when analyzing large datasets with many experiments


