crime_assault = df[df["offence"].str.contains('ASSAULT', flags=re.IGNORECASE, regex=True)] #Step 1.

values_assault = crime_assault["offence"].value_counts() #Step 2.

for key,value in values_assault.iteritems(): #Step 3.
    if value < 1500:
       values_assault= pd.DataFrame(values_assault.rename({key: "other"}))


values_assault=values_assault.groupby(values_assault.index).sum().sort_values("offence",ascending=False) #Step 4.

labels_assault = values_assault.index #Step 5.