# growth_rates

This repository is for quickly determining microbial growth rates in high-throughput from plate reader OD600 data. It uses the iPython (Jupyter Notebooks) environment to import raw plate reader data from an Excel export, find a best-fit linear regression to Time (x) v. ln(OD600) (y) data, quality check fits, and export fit parameters back to Excel for easy end-user data manipulation and visualization. To start, make sure you have installed **Jupyter Notebooks** or **JupyterLab**. I recommend using [Anaconda Navigator](https://www.anaconda.com/download) to download and access these tools, as Anaconda will make installing package dependencies (see below) much easier.


## Scientific background
Bacteria and other microorganisms grown in liquid culture is typically modeled in four sequential phases: 
1. **lag phase**, where microbes are acclimated to being introduced to a new environment and therefore do not grow quickly,
2. **exponential phase**, where the microbial population undergoes rapid expansion limiting only by the organism's growth rate,
3. **stationary phase**, where the culture is now so dense that microbes must compete for limited resources and slow their growth to increase long-term chances of survival, and
4. **death phase**, where nutrients are depleted to the point that the culture experiences a large-scale die-off.

Of these four phases, *we are typically interested in modeling exponential phase*, as this can provide insight into the organism's fitness under optimal culture conditions with no other competing factors.

Shown in the image below you can see the first three phases of the growth of bacteria in liquid culture: exponential phase has been modeled by a black curve superimposed on the blue raw data values, with lag phase preceding and stationary phase following exponential phase. growth_rates generated the black model curve as the best overall fit to the blue data curve.

![image](https://github.com/user-attachments/assets/f7947a17-3b5c-4a5b-89e1-7dd6dc9a4f9f)


## Statistical regression analysis and window picking

In order to find the best model of exponential phase, the "raw" OD600 data is first log-transformed, generating a linear relationship between time and ln(OD600) for the exponential phase that can be modeled using the LinearRegression tool in scikit-learn. For example, here is the log-transform of the graph from above with a linear best fit:

![image](https://github.com/user-attachments/assets/171f1be0-2f78-4f0b-953e-ae8f1f450f7c)

However, this relationship is only linear for a short time during exponential phase, so this program applies a sliding window algorithm to identify the best fit during exponential phase. This algorithm attempts linear regressions across the entire range of data at various window "sizes", then selects the best fit by maximizing the coefficient of determination (R^2) of the fitted line. To best model data, several further constraints are included with exact values based on trial-and-error, but these values may need to be adjusted with differing culture conditions:
1. OD600 values are forced to increase at least ~4.5-fold over the time window to filter out non-exponential phase fits.
2. OD600 value at the end of the fitted time window must be at least ~0.05 to filter out fits from noise at low absorbances.
3. Best-fit R^2 value must be >0.99 to classify as a valid fit to filter out fitting to subpar data.
4. Fit window must must include at least 100 minutes of culture time and 8 measurements.

The code determining all of these constraints can be found in the "well_fitter" function and its "find_linear_regime" sub-function.

## Sampling, variables, and naming
This pipeline was developed for utility of bench-facing scientists in mind: in order to designate different samples, blank wells, and wells to drop from analysis, they simply need to designate variable attributes in a second sheet on their data file with column labels designated in row 2 (for reference, check the associated example_data). Based on these conventions, the program automatically identifies replicates, background subtracts samples using the blank wells, and IDs samples according to their well and collected variable attributes. For example:


![image](https://github.com/user-attachments/assets/eb9fbf5e-66eb-46b0-8603-df2f1f6b400d)


**Required column labels** (order unimportant):
* "Well": These should be within the set of alphanumerics created from the product of (A-P)(1-24), e.g., "A24", "F6", "P15". The allowed set includes all wells on a standard 96- or 384-well plate.
* "Sample": This column is essential because it is where blank and dropped wells should be designated. **Wells labeled "blank" will be used for background subtraction;** the program requires at least one blank well per media type per experiment to run. By default, the program will also attempt to check all blank wells for contamination via a statistical method compared to other blank well timepoints and/or crossing an OD600 threshold of 0.5. Contamination-checking behavior can be turned off within the data_cleaner function. **Wells labeled "drop" are designated as manually excluded from analysis and ignored.** Otherwise, variable attributes in this column are treated as the same as variable attributes in other columns for the purpose of grouping.
* "Media": This column is essential because it is used to group blank wells during background subtraction. Setting all values in this column to the same value is allowed and results in background subtraction being performed once on the entire plate.

As many other non-essential column labels and attributes as desired can be designated, with analysis automatically using variable attributes from all columns except "Well" to identify sample replicates. If multiple plates/experiments are being analyzed at once, it is recommended to ensure that column labels are consistent across all plates.

**DO NOT DESIGNATE** column labels as any of the following, as these labels are generated and used internally by the program:\
 ["Assay", "Temperature", "Value", "Experiment"]


## Path and directory (folder) setup
Make sure this script is saved in the same folder as a folder labeled 'xlsx' containing your plate reader data files saved with the .xlsx extension.

Additionally, ***you will need to modify the file_target variable to a string that is contained in the data files you want to analyze.*** Multiple data files can be analyzed at once if they all contain this same string. Note that these files will each be assigned a unique label by file name under the 'Experiment' column in the resulting dataframe.

For example, using file_target 'expt' will find and analyze '/expt1.xlsx', '/expt2.xlsx', and '/expt3.xlsx' in the 'xlsx' folder together, and the script will tag each of the resulting subdatasets as 'expt1', 'expt2', and 'expt3' respectively.


## Outputs

The outputs of this script are two Excel files, "Best fits by well.xlsx" and "Mean fits by grouped replicates.xlsx". This is done so that these final parameter values can be plugged in to the graphing / visualization tool of your choice.

"Best fits by well.xlsx" displays best fit parameters for each individual well:
* Growth rate (/hr) is the culture's derived growth rate.
* logT Y-intercept is the Y-intercept of the linear equation of best fit on log-transformed y-data. This can usually be ignored.
* R2 is the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).
* XTimeStart and XTimeStop are the start and stop time value in hours where the best fit window was found.

![image](https://github.com/user-attachments/assets/972ec51b-9cd1-4d5b-953f-bc84300de2ea)

"Mean fits by grouped replicates.xlsx" displays the mean and standard deviation of the above statistics across curves identified as replicates. It also shows which wells are included in the replicate group, displays how many replicates generated a valid fit for each sample, and derives a value for doubling time from the average growth rate.

![image](https://github.com/user-attachments/assets/3c67e77d-ba83-42c1-8893-c1e48ee32d60)


### Note on example data
Sometimes in science (especially when working with tiny bugs with messy computers for brains!) things don't go as planned. This dataset is messy, with many missing or poor growth curves, which has resulted in many gaps in our final parameter fits as seen above. While these data represent a disappointing day for a scientist, they comprise a great training dataset to use when building this script to ensure that messy data doesn't trip up analysis and that fitting parameters are tuned well! Hopefully, all of the datasets you analyze with this script will look better than this example. :)

## Extensibility
One of the primary motivations for this script is the ability to automate analysis of growth rate data in high-throughput. As such, it's designed to accept and analyze as many data files as you'd like to give it at once. For best results, make sure variable column labels match across all analyzed files.

## Dependencies
growth_rates has the following package dependencies, which are likely pre-installed if using Jupyter Notebooks via Anaconda:
pandas, numpy, scipy, math, glob, re, warnings, os, pathlib, time, functools

## To-do
### Possible future updates:
* Providing additional import functionality for a range of Excel raw data formats
* Adding API support for plugging into larger experimental workflows that use growth parameters for model training
* More graphical outputs
* Small updates that continue to make script more user-friendly and accessible to non-programmers
* Adding support for fitting growth data to the Gompertz curve
* More metadata functionality when analyzing large datasets with many experiments

## Contact
I love talking about science and hearing that people are using my work! If you have any questions about how to use growth_rates or suggestions on how to make it better, feel free to connect with me on LinkedIn at https://www.linkedin.com/in/samgowland/


