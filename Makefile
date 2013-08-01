
all:

generatedb:
	(cd src; $(MAKE) "CFLAGS=-O3" clean all)
	./src/kocr images/numbers/list-num.lst
	cp -p images/numbers/list-num.db databases
	./src/kocr images/mbsdb/list-mbs.lst
	cp -p images/mbsdb/list-mbs.db databases
	./src/kocr images/sample-ocrb/list-ocrb.lst
	cp -p images/sample-ocrb/list-ocrb.db databases
	(cd images/numocrb; rm ./*;\
		ln -s ../numbers/*.png .; ln -s ../sample-ocrb/*.png .;\
		cp -p ../numbers/list-num.lst list-numocrb.lst;\
		cat ../sample-ocrb/list-ocrb.lst >> list-numocrb.lst; )
	./src/kocr images/numocrb/list-numocrb.lst
	cp -p images/numocrb/list-numocrb.db databases

	(cd src; $(MAKE) "CFLAGS=-O3 -DUSE_SVM" clean all)
	./src/kocr images/numbers/list-num.lst
	cp -p images/numbers/list-num.xml databases
	./src/kocr images/mbsdb/list-mbs.lst
	cp -p images/mbsdb/list-mbs.xml databases
	./src/kocr images/sample-ocrb/list-ocrb.lst
	cp -p images/sample-ocrb/list-ocrb.xml databases
	(cd images/numocrb; rm ./*.png ./*.db ./*.xml ./*.lst;\
		ln -s ../numbers/*.png .; ln -s ../sample-ocrb/*.png .;\
		cp -p ../numbers/list-num.lst list-numocrb.lst;\
		cat ../sample-ocrb/list-ocrb.lst >> list-numocrb.lst; )
	./src/kocr images/numocrb/list-numocrb.lst
	cp -p images/numocrb/list-numocrb.xml databases
