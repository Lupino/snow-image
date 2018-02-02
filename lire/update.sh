#!/usr/bin/env bash

find -name 'batch_*.xml' | while read F;do
    curl -XPOST -F file=@$F 'http://liresolr.huabot.com/solr/lire/update?commitWithin=3000&boost=1.0&overwrite=true&wt=json'
done
