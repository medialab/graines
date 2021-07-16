import pandas as pd
from tqdm import tqdm
from pymed import PubMed

#query = '"information science"[MeSH Terms] AND "neoplasms"[MeSH Terms] AND 1997/01/01:2030/01/01[dp]'

def pubmed_query(query, destination_path, max_results=100):

	pubmed = PubMed(tool="MyTool", email="charles.dedampierre@sciencespo.fr")

	results = pubmed.query(query, max_results=max_results)

	articleList = []
	articleInfo = []

	for article in results:
	# Print the type of object we've found (can be either PubMedBookArticle or PubMedArticle).
	# We need to convert it to dictionary with available function
	    articleDict = article.toDict()
	    articleList.append(articleDict)

	# Generate list of dict records which will hold all article details that could be fetch from PUBMED API
	for article in articleList:
	#Sometimes article['pubmed_id'] contains list separated with comma - take first pubmedId in that list - thats article pubmedId
	    pubmedId = article['pubmed_id'].partition('\n')[0]
	    # Append article info to dictionary 
	    articleInfo.append({u'pubmed_id':pubmedId,
	                       u'title':article['title'],
	                       u'keywords':article['keywords'],
	                       u'journal':article['journal'],
	                       u'abstract':article['abstract'],
	                       u'conclusions':article['conclusions'],
	                       u'methods':article['methods'],
	                       u'results': article['results'],
	                       u'copyrights':article['copyrights'],
	                       u'doi':article['doi'],
	                       u'publication_date':article['publication_date'], 
	                       u'authors':article['authors']})
	    
	# Generate Pandas DataFrame from list of dictionaries
	articlesPD = pd.DataFrame.from_dict(articleInfo)
	articlesPD.to_csv(destination_path + 'pubmed_query_max_results_{}.gzip'.format(max_results), compression='gzip')
	print("It is done!")


