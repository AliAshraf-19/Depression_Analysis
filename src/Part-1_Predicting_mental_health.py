import pandas as pd
import numpy as np
from scipy.stats import kruskal
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


import statsmodels.api as smm
from statsmodels.formula.api import ols

from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import Table

df= pd.read_csv("C:/Users/Downloads/depression_data.csv")

df.columns

duplicate = df[df.duplicated()]

print(f"Duplicate Rows : {duplicate}" )

df_num = df.select_dtypes(['int64','float64'])
correlation_matrix = df_num.corr()

fig = px.imshow(correlation_matrix, text_auto=True)
fig.update_layout(title_text="Heatmap for the Numerical data")
fig.show()

for column in df_num.columns:
      yes = df[df["History of Mental Illness"]=="Yes"][column]
  no  = df[df["History of Mental Illness"]=="No"][column]
  p = kruskal(yes, no)
  print(f"The significance score for {column} and Mental Illness",p.pvalue)
  
categorical= df.select_dtypes(include=['object','category']).drop(columns=["History of Mental Illness"])
ordinal_columns =['Education Level','Physical Activity Level','Alcohol Consumption','Dietary Habits']

for column in categorical.columns:
    if column not in ordinal_columns:
        data = pd.crosstab(df[column],df['History of Mental Illness'])
        contingency_table = Table(data.values)
        chi_square_test = contingency_table.test_nominal_association()
        print(f"The p-value CHI score for {column} and Mental Illness",chi_square_test)
    else:
        data = pd.crosstab(df[column], df['History of Mental Illness'])
        contingency_table = Table(data.values)
        chi_square_test = contingency_table.test_ordinal_association()
        print(f"Chi-Square Ordinal for {column} and Mental Illness: {chi_square_test}")
        
data = df.rename(columns={"Marital Status": "Marital_Status","Education Level": "Education_Level"})
moore_lm = ols('Income ~ C(Marital_Status, Sum)*C(Education_Level, Sum)', data=data).fit()

table = smm.stats.anova_lm(moore_lm, typ=2)
table

dfp = df.groupby(['History of Mental Illness'])['History of Mental Illness'].count().reset_index(name='count')

fig = px.pie(dfp, values='count', names='History of Mental Illness')
fig.update_layout(title_text='Pie chart for History of Mental Illness ', title_x=0.5)
fig.show()

fig = px.violin(df, x='Education Level', y="Income", points='all', color='History of Mental Illness')
fig.update_layout(
    title=dict(text="Income Distribution around Education focusing Mental Health", font=dict(size=15)))
fig.show()

fig = px.histogram(df, x="Smoking Status", color="History of Mental Illness") #histnorm='percent',
fig.update_layout(
    title=dict(text="Income Distribution around Mental Health", font=dict(size=15)))
fig.show()

fig = px.histogram(df, x="Marital Status",  color="History of Mental Illness") #histnorm='percent',
fig.update_layout(
    title=dict(text="Marital Status Distribution around Mental Health", font=dict(size=15)))
fig.show()

df1 =  df.copy()
df1['Age_bin'] = pd.cut(df1['Age'], bins=[-np.inf, 18, 35, 50, 65, np.inf], labels=['Under 18', '18-35', '36-50', '51-65', '65+'])

fig = px.histogram(df1, x="Age_bin",  color="History of Mental Illness") #histnorm='percent',
fig.update_layout(
    title=dict(text="Age Distribution around Mental Health", font=dict(size=15)))
fig.show()

df1['Income_bin'] = pd.qcut(df1['Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

fig = px.histogram(df1, x="Income_bin",  color="History of Mental Illness") #histnorm='percent',
fig.update_layout(
    title=dict(text="Income Distribution around Mental Health", font=dict(size=15)))
fig.show()

fig1 = px.sunburst( df, path=['History of Mental Illness', 'Physical Activity Level'])

fig2 = px.sunburst( df, path=['History of Mental Illness', 'Education Level'])

fig3 = px.sunburst( df, path=['History of Mental Illness', 'Alcohol Consumption'])

fig4 = px.sunburst( df, path=['History of Mental Illness', 'Smoking Status'])

fig5 = px.sunburst( df, path=['History of Mental Illness', 'Sleep Patterns'])

fig6 = px.sunburst( df, path=['History of Mental Illness', 'Dietary Habits'])

fig = make_subplots(rows=3, cols=2, specs=[
    [{"type": "sunburst"}, {"type": "sunburst"}], [{"type": "sunburst"}, {"type": "sunburst"}],[{"type": "sunburst"}, {"type": "sunburst"}]
], subplot_titles=("Physical Activity", "Education Level", "Alcohol Consumption","Smoking Status", "Sleeping Pattern", "Dietary Habit"))

fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)
fig.add_trace(fig3.data[0], row=2, col=1)
fig.add_trace(fig4.data[0], row=2, col=2)
fig.add_trace(fig5.data[0], row=3, col=1)
fig.add_trace(fig6.data[0], row=3, col=2)
fig.update_layout(height=900, width=700,
                  title_text="Multiple Subplots for Mental illness causation")

fig.show()