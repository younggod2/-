Одни и те же структуры в AB_Bind, SKEMPI2_PDBs, Protein Data Bank отличаются друг от друга, поэтому когда извлекались фичи, в первом приоритете были .pdb из AB_Bind, во втором — SKEMPI2_PDBs

AB_Bind источник: https://github.com/sarahsirin/AB-Bind-Database + pdb_selaltloc.py прогнал по ним
SKEMPI2_PDBs источник: https://life.bsc.es/pid/skempi2/database/index

pdb_selaltloc.py (https://www.bonvinlab.org/pdb-tools/) скрипт оставляет наиболее вероятный конформер аминокислот в .pdb, применил на AB_Bind

В skempi_v2.csv заменил разделитель ";" на ","
