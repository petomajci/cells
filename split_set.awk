BEGIN{

while (getline< SIRNAlist)
  ss[$1]=1
}

{
  if (NR==1) print
	if(ss[$5]==1) print
}
