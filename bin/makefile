vpath %.cpp ../src/
vpath %.h ../include/
VPATH = ../src/:../res/

.PHONY: clean
.PHONY: create_index
.PHONY: demo_restaurant
.PHONY: demo_laptop
.PHONY: training_restaurant
.PHONY: training_laptop
.PHONY: test_restaurant
.PHONY: test_laptop
.PHONY: compile
compile:
	cd ../src; g++ $(filename) ../lib/olsrnn.so  -o ../bin/$(patsubst %.cpp,%,$(filename))
demo_restaurant: main ../res/Word_Vector_Index
	./main demo_restaurant
demo_laptop: main ../res/Word_Vector_Index
	./main demo_laptop
test_restaurant: main ../res/Word_Vector_Index
	./main test_restaurant $(model_file)
test_laptop: main ../res/Word_Vector_Index
	./main test_laptop $(model_file)
training_restaurant: main ../res/Word_Vector_Index
	./main training_restaurant $(learning_rate) $(momentum_rate) $(regularization) $(interation_num)
training_laptop: main ../res/Word_Vector_Index
	./main training_laptop $(learning_rate) $(momentum_rate) $(regularization) $(interation_num)
create_index: main
	./main create_index $(file_name)	
object = main.o aspect_tool.o olsrnn.o utility.o word2vec_tool.o string_tool.o
main: $(object)
	cd ../src;g++ -o ../bin/main $(object) 
main.o: main.cpp aspect_tool.h olsrnn.h utility.h word2vec_tool.h string_tool.h
	cd ../src; g++ -c main.cpp
aspect_tool.o: aspect_tool.cpp aspect_tool.h olsrnn.h word2vec_tool.h string_tool.h
	cd ../src; g++ -c aspect_tool.cpp
olsrnn.o: olsrnn.cpp olsrnn.h utility.h
	cd ../src; g++ -c olsrnn.cpp
utility.o: utility.cpp utility.h
	cd ../src; g++ -c utility.cpp
word2vec_tool.o: word2vec_tool.cpp word2vec_tool.h string_tool.h
	cd ../src; g++ -c word2vec_tool.cpp
string_tool.o: string_tool.cpp string_tool.h
	cd ../src; g++ -c string_tool.cpp
clean:
	cd ../src; rm *.o main
