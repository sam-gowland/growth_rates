This is a guide to get growth_rates running as quickly and simply as possible on your local machine.

## 1. Make sure you have Jupyter Notebooks and/or JupyterLab installed.


## 2. Save the growth_rates script in the same directory (folder) as a folder named 'xlsx' that your plate-reader data exported to Excel .xlsx format.


![image](https://github.com/user-attachments/assets/9cc1300b-d98c-44fa-a9d6-1c8e50a723f2)


## 3. Define variables on Sheet 2 of your data file.
Be sure to include columns labeled "Well", "Sample", and "Media" and DO NOT include columns labeled "Assay", "Temperature", "Value", or "Experiment".

In the sample column, use 'blank' for blank wells and 'drop' to ignore wells. 


![image](https://github.com/user-attachments/assets/eb9fbf5e-66eb-46b0-8603-df2f1f6b400d)


## 4. Open the growth_rates script in Jupyter Notebooks or JupyterLab and modify the 'file_target' variable to a string uniquely contained by the files in the "../xlsx/" folder that you want to analyze.


![image](https://github.com/user-attachments/assets/da872eb3-72fb-43bd-96d0-4e564af42fae)



## 5. Run the script. Make sure to run both the first cell and second cell. Your results will be output to Excel files stored in the '/results/(your file_target)/' folder.


![image](https://github.com/user-attachments/assets/1611daa3-1ce0-45bd-852d-16659d37083d)



![image](https://github.com/user-attachments/assets/7040ef93-028e-402e-8c73-74d48ede4b3c)

