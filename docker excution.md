# tensorflow-tutorial
tensorflow tutorial manual for my labtop.
([Reference1](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/get_started/os_setup.html) /  [Reference2](https://gist.github.com/haje01/202ac276bace4b25dd3f)) 

1. Start a "Docker Quickstart Terminal". 

	>Docker terminal in Windows. (dir command shows a windows' dir.)
	
	>'$ docker ps' show running containers
	
	>'$ docker images' show list of images
2. Command   `$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-devel`

	> 8888 is a port for Jupyter notebook.
	
	> '~~winpty~~' is not necessary.
	
	> ~~b.gcr.io/tensorflow/tensorflow-full~~ is not contaion some examples (ex. mnist)
	
	> If you do not add ':latest-devel', server problem may occur.
3. 



