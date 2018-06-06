all: fnbod.so fbrusselator.so

fnbod.so: fnbod.f
	f2py -c fnbod.f -m fnbod

fbrusselator.so: fbrusselator.f
	f2py -c fbrusselator.f -m fbrusselator

clean:
	rm fnbod.so
	rm fbrusselator.so
