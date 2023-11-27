VAR=' Clean the first caches'
VAR1='info about cleaning'
VAR2='after python execution clean caches'
VAR3='creat a new empty folder'
VAR4='delet a figs folder'

execute:
	$(info Caches have been deleted)

middle:execute
	@rm -r __pycache__

pyccel_cleaning :middle
	@rm -r __epyccel__

make_figs :pyccel_cleaning
	@rm -r figs
	@mkdir figs

clean_all:make_figs
	$(info All figures have been deleted, and a new folder named figs has been created )
