(* ::Package:: *)

(*The number of sites*)
nSite=10;
(*Pauli Matrices*)
sigma={({
 {1., 0.},
 {0., 1.}
}),({
 {0., 1.},
 {1., 0.}
}),({
 {0., -I},
 {I, 0.}
}),({
 {1., 0.},
 {0., -1.}
})};
sx=SparseArray/@Table[KroneckerProduct@@Table[If[j==i,sigma[[2]],sigma[[1]]],{j,1,nSite}],{i,1,nSite}];
sy=SparseArray/@Table[KroneckerProduct@@Table[If[j==i,sigma[[3]],sigma[[1]]],{j,1,nSite}],{i,1,nSite}];
sz=SparseArray/@Table[KroneckerProduct@@Table[If[j==i,sigma[[4]],sigma[[1]]],{j,1,nSite}],{i,1,nSite}];
(*1 means spin up*)
binToVec[bin_,num_]:=SparseArray[{FromDigits[1-bin,2]+1->1.},{2^num},0.];
(*Fill zeros at the position in ord to tar*)
binFill[tar_,ord_]:=Catch[Block[{res={},taru=tar},Do[If[MemberQ[ord,i]||Length[taru]==0,AppendTo[res,0.],AppendTo[res,taru[[1]]];taru=Drop[taru,1]],{i,1,nSite}];Throw[res]]];
(*Insert obj into tar with the order of ord*)
(*binIns[{1,2,3},{a,b,c,d},{2,3,5}]={a,1,2,b,3,c,d}*)
binIns[obj_,tar_,ord_]:=ReplacePart[binFill[tar,ord],Table[ord[[i]]->obj[[i]],{i,1,Length[ord]}]];
(*Partial Trace of matrix mat except sites in list of sites*)
partialTr[mat_,sites_]:=Catch[Block[{nLeft=Length[sites],trLis,ntrLis},
trLis=PadLeft[Table[IntegerDigits[k,2],{k,0,2^(nSite-nLeft)-1}]];
ntrLis=PadLeft[Table[IntegerDigits[k,2],{k,0,2^nLeft-1}]];
Throw[Table[Sum[binToVec[binIns[ntrLis[[i]],trLis[[k]],sites],nSite].mat.binToVec[binIns[ntrLis[[j]],trLis[[k]],sites],nSite],{k,1,Length[trLis]}],{i,1,Length[ntrLis]},{j,1,Length[ntrLis]}]]
]];
hellingDis[p1_,p2_]:=Re[Sqrt[1-Sum[Sqrt[p1[[i]]p2[[i]]],{i,1,Length[p1]}]]];
klDis[p1_,p2_]:=Re[p1.(If[#==0,0,Log[#]]&/@(p1))-p1.(If[#==0,0,Log[#]]&/@p2)];
ent[rho_]:=-Re[Total[If[Abs[#]<=10^-6,0.,# Log[#]]&/@Eigenvalues[rho]]];
(*Correlation of the state on two sites*)
corr[rho_,sitei_,sitej_]:=Catch[Block[{p1=Diagonal[partialTr[rho,{sitei}]],p2=Diagonal[partialTr[rho,{sitej}]],p12=Diagonal[partialTr[rho,{sitei,sitej}]],p1p2},
p1p2=Flatten[KroneckerProduct[p1,p2]];
Throw[hellingDis[p1p2,p12]]
]];
(*Correlation of Everett III*)
corrEv[rho_,sitei_,sitej_]:=Catch[Block[{p1=Diagonal[partialTr[rho,{sitei}]],p2=Diagonal[partialTr[rho,{sitej}]],p12=Diagonal[partialTr[rho,{sitei,sitej}]],p1p2},
p1p2=Flatten[KroneckerProduct[p1,p2]];
Throw[klDis[p12,p1p2]]
]];
(*Density Matrix distance*)
trDis[op1_,op2_]:=Re[0.5*Total[Sqrt/@Eigenvalues[ConjugateTranspose[op1-op2].(op1-op2)]]];
dmDis[rho_,sitei_,sitej_]:=
Catch[Block[{p1=partialTr[rho,{sitei}],p2=partialTr[rho,{sitej}],p12=partialTr[rho,{sitei,sitej}],p1p2},
p1p2=KroneckerProduct[p1,p2];
Throw[trDis[p12,p1p2]]
]];
miDis[rho_,sitei_,sitej_]:=Catch[Block[{p1=partialTr[rho,{sitei}],p2=partialTr[rho,{sitej}],p12=partialTr[rho,{sitei,sitej}]},
Throw[ent[p1]+ent[p2]-ent[p12]]
]];
(*Calculate the von Neumann entropy on state psi seperated at site*)
vNent[psi_,site_]:=-Re[Total[If[#==0,0,# Log[#]]&/@Eigenvalues[partialTr[KroneckerProduct[psi,Conjugate[psi]],Array[#&,site]]]]];
(*Find the first time of element in lis is greater than th, with times step dt*)
hittime[lis_,th_,dt_]:=(FirstPosition[lis,n_/;n>=th] dt)[[1]];
(*Commutator of two operators*)
commt[op1_,op2_]:=op1.op2-op2.op1;
