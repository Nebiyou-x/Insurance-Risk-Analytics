import matplotlib.pyplot as plt
prov = loss_by_province.reset_index()
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(prov['province'], prov['loss_ratio'], alpha=0.7)
# overlay bubble: total premium -> scaled
sizes = (prov['total_premium'] / prov['total_premium'].max()) * 1000
ax.scatter(prov['province'], prov['loss_ratio'], s=sizes, alpha=0.4)
ax.set_ylabel('Loss ratio')
ax.set_title('Loss ratio by Province (bubble size = total premium)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/lossratio_by_province.png', dpi=150)

import seaborn as sns
# pivot table for top zipcodes by premium
top_zip = df.groupby('postalcode').totalpremium.sum().nlargest(30).index
tmp = df[df.postalcode.isin(top_zip)].groupby(['postalcode', 'month']).apply(lambda g: g.totalclaims.sum()/g.totalpremium.sum()).reset_index(name='loss_ratio')
pivot = tmp.pivot(index='postalcode', columns='month', values='loss_ratio').fillna(0)
plt.figure(figsize=(14,9))
sns.heatmap(pivot, cmap='viridis', linewidths=0.2)
plt.title('Loss ratio per month for top 30 postal codes')
plt.savefig('reports/figures/heatmap_zip_month.png', dpi=150)

top_makes = df.groupby('make').totalclaims.sum().nlargest(20).index
plt.figure(figsize=(14,7))
sns.boxplot(x='make', y='totalclaims', data=df[df.make.isin(top_makes)])
plt.yscale('symlog')  # or log scale to manage heavy tails
plt.title('Claim amounts distribution by Vehicle Make (top 20 makes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/claims_by_make_boxplot.png', dpi=150)
