# Internet Of Things (IoT) Class Project
IoT project to aggregate weather data using an ESP32 and analyze the data set to identify any correlations.



As a team of 2, we gathered climate data using an ESP32 and analyzed the dataset using python pandas and Matplotlib
Created prediction models and charts to illustrate data correlation

In details:

* Component 1:
- Data was collected every 2 minutes for 2hrs each day. Totally to 10 hours’ worth of data. The data was then stored on google sheets.
- I utilized IFTTT to send the data to the google sheets and update it in real time.
  - A python script was written to connect to the Wi-Fi, and then sent a post request to IFTTT, which updated the google sheets with the current live temperature and humidity data from the ESP32. Each reading had a timestamp associated with it.
 
 * Component 2:
 - The data was then analyzed and visualized in python. 
  - First, I extracted the data from the csv file using the python pandas library, and then the date was parsed in order to have it in the right format.
  - The data for each respective day was extracted and then I then plotted the data on a graph for a visual representation.
 
 * Component 3:
 * A linear regression model was then made and visualized using the data gathered from the previous steps. This was to highlight the relationship between the variables, particulary between temperature and time, and between humidity and time.
 
