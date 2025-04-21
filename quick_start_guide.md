This is a guide to get growth_rates running as quickly and simply as possible on your local machine.

## 1. Save the growth_rates script in the same directory (folder) as a folder named 'xlsx' that your plate-reader data exported to Excel .xlsx format.


![image](https://github.com/user-attachments/assets/9cc1300b-d98c-44fa-a9d6-1c8e50a723f2)


## 2. Define variables on Sheet 2 of your data file.
Be sure to include columns labeled "Well", "Sample", and "Media" and DO NOT include columns labeled "Assay", "Temperature", "Value", or "Experiment".

In the sample column, use 'blank' for blank wells and 'drop' to ignore wells. 


![image](https://github.com/user-attachments/assets/eb9fbf5e-66eb-46b0-8603-df2f1f6b400d)



## 3. Run the script. The program will ask for a file-targeting string to specify which files in the 'xlsx' folder to analyze (returning without entering any characters will analyze all available files). If running the .ipynb file, make sure to run both the first cell and second cell. Your results will be output to Excel files stored in the '/results/(your file_target)/' folder.


![image](https://github.com/user-attachments/assets/1611daa3-1ce0-45bd-852d-16659d37083d)



![image](https://github.com/user-attachments/assets/7040ef93-028e-402e-8c73-74d48ede4b3c)

