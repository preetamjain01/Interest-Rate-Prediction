FROM python:3



# install additional python packages
RUN pip3 install ipython
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install seaborn

# Bundle app source
COPY loan.csv /usr/local/bin/loan.csv
COPY LCDataDictionary.xlsx /usr/local/bin/LCDataDictionary.xlsx
COPY Predicting_Interest_rate_LendingClub.py /usr/local/bin/Predicting_Interest_rate_LendingClub.py

ADD Predicting_Interest_rate_LendingClub.py .
CMD ["python", "/usr/local/bin/Predicting_Interest_rate_LendingClub.py"]