#!/usr/bin/env bash

find -name '*.jpg' > infile.txt

java -cp lib/commons-codec-1.11.jar:lib/lire.jar:lib/liresolr.jar net.semanticmetadata.lire.solr.indexing.ParallelSolrIndexer -i infile.txt -y ce,ac -o output.xml
