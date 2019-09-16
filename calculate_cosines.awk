BEGIN{
  eps=0.000001
}


{ 
  N1=0 
  C1=0 
  cos1=0

  N2=0  
  C2=0  
  cos2=0
  cos3=0
  cos4=0 
  
  for(i=1;i<=1024;i++){ 
     f1[i]=$i 
     c1[i]=$(i+2048) 
     
     
     f2[i]=$(i+1024)  
     c2[i]=$(i+3072)


     N1+=f1[i]*f1[i] 
     N2+=f2[i]*f2[i]
     C1+=c1[i]*c1[i]
     C2+=c2[i]*c2[i]
 
     cos1+=c1[i]*f1[i]
     cos2+=c2[i]*f1[i] 
     cos3+=c1[i]*f2[i]
     cos4+=c2[i]*f2[i] 
  }

  N1=sqrt(N1)
  C1=sqrt(C1)
  N2=sqrt(N2)
  C2=sqrt(C2) 
  print cos1/(N1*C1+eps)" "cos2/(N1*C2+eps)" "cos3/(N2*C1+eps)" "cos4/(N2*C2+eps) 
}
