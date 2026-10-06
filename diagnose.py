from search import search

results = search("What was the net income for the year?",
                 company="ACRES Commercial Realty Corp.", k=50)

print("Chunks containing 'net income' within the top 50:")
for rank, (doc, meta, dist) in enumerate(results, 1):
    if "net income" in doc.lower():
        body = doc.split("\n\n", 1)[-1][:120].replace("\n", " ")
        print(f"  rank {rank:2} | page {meta['page']:3} | dist {dist:.3f} | {body}")