PREFIX=/usr/local

all:

generatedb: gen-nn-db gen-svm-db

gen-nn-db:
	(cd src; $(MAKE) "SOLVER=" clean all)
	./src/kocr images/numbers/list-num.lst
	cp -p images/numbers/list-num.db databases
	./src/kocr images/mbsdb/list-mbs.lst
	cp -p images/mbsdb/list-mbs.db databases
	(cd images/mbscdb; rm ./*.db ./*.xml ;\
		ln -s ../mbsdb/*.png .; \
		cat ../mbsdb/list-mbs.lst >> list-mbsc.lst; )
	./src/kocr images/mbscdb/list-mbsc.lst
	cp -p images/mbscdb/list-mbsc.db databases
	(cd images/mbsczdb; rm ./*.db ./*.xml ;\
		ln -s ../mbscdb/*.png .; \
		cat ../mbscdb/list-mbsc.lst >> list-mbscz.lst; \
		ln -s ../plus-minus/*.png .; \
		cat ../plus-minus/list-plus-minus.lst >> list-mbscz.lst; )
	./src/kocr images/mbsczdb/list-mbscz.lst
	cp -p images/mbsczdb/list-mbscz.db databases
	./src/kocr images/sample-ocrb/list-ocrb.lst
	cp -p images/sample-ocrb/list-ocrb.db databases
	(cd images/numocrb; rm ./*;\
		ln -s ../numbers/*.png .; ln -s ../sample-ocrb/*.png .;\
		cp -p ../numbers/list-num.lst list-numocrb.lst;\
		cat ../sample-ocrb/list-ocrb.lst >> list-numocrb.lst; )
	./src/kocr images/numocrb/list-numocrb.lst
	cp -p images/numocrb/list-numocrb.db databases

gen-svm-db:
	(cd src; $(MAKE) "SOLVER=SVM" clean all)
	./src/kocr images/numbers/list-num.lst
	cp -p images/numbers/list-num.xml databases
	./src/kocr images/mbsdb/list-mbs.lst
	cp -p images/mbsdb/list-mbs.xml databases
	./src/kocr images/mbscdb/list-mbsc.lst
	cp -p images/mbscdb/list-mbsc.xml databases
	./src/kocr images/mbsczdb/list-mbscz.lst
	cp -p images/mbsczdb/list-mbscz.xml databases
	./src/kocr images/sample-ocrb/list-ocrb.lst
	cp -p images/sample-ocrb/list-ocrb.xml databases
	(cd images/numocrb; rm ./*.png ./*.db ./*.xml ./*.lst;\
		ln -s ../numbers/*.png .; ln -s ../sample-ocrb/*.png .;\
		cp -p ../numbers/list-num.lst list-numocrb.lst;\
		cat ../sample-ocrb/list-ocrb.lst >> list-numocrb.lst; )
	./src/kocr images/numocrb/list-numocrb.lst
	cp -p images/numocrb/list-numocrb.xml databases

install-db:
	(dir=share/kocr/databases; if [ ! -e $$dir ]; then mkdir -p $(PREFIX)/$$dir; fi)
	install -c -m 444 -o root -g root databases/* $(PREFIX)/share/kocr/databases
