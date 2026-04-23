#variant 1 - native squai context extraction (very simple word overlap)

#variant 2 - Biencoder (selects top 1) out of floating windows of up to 5 sentences
#variant 3 - BM25 selects top 1 out of floating window up to 5 sentences

#variant 4 - Biencoder selects top 10 (out of floating windows), then BM25 selects top 1                -maybe a bit trash
#variant 5 - BM25 selects top 10 (out of floating windows), then Biencoder selects top 1

#variant 6 - Biencoder selects top 10 (out of floating windows) top 10, cross encoder selects top 1
#variant 7 - BM25 selects top 10 (out of floating windows) top 10, cross encoder selects top 1

#variant 8 - Biencoder and bm25 select top 1 (using RRF)

#variant 9 - BiEncoder and keyword bm25 select top 10 together, cross encoder selects top 1

# variant 10 - extract the context using LLM (prompt to extract the best fit)


#--> select top 3 performing versions, add decontextualization option to them