# Processing wikidumps

```bash
wget -r -np -nH --cut-dirs=2 -R "index.html*" --accept-regex="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles[0-9]+.xml.*bz2\$" https://dumps.wikimedia.org/enwiki/latest/
```

```bash
find /Users/AKB/GitHub/nonce2vec/data/wikipedia/20180920 -type f -name "*.bz2" | xargs -n1 -I file pbzip2 -p55 -d file
```
