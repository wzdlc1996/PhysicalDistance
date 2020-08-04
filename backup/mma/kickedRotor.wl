(* ::Package:: *)

sparsBesJ[n_,x_]:=If[n>10.*x,0.,Chop[BesselJ[n,x]]];
SetAttributes[sparsBesJ,Listable];

flcUEle[s1_,s2_]:=(-I)^(s2-s1)BesselJ[s2-s1,k/hbar]Exp[-I s1^2 hbar/2.];
blcEle[s1_,s2_]:=Sum[flcUEle[s1,s2+m^2 l],{l,-Floor[6. k/\[Pi]],Floor[6. k/\[Pi]]}];

dist[x_,y_]:=Catch[
Block[
{diff=Abs[x-y]},
Throw[Total[Min/@Transpose[{diff,2.*\[Pi]-diff}]]];
]
];

phi2Ph[phi_]:=Catch[
Block[
{usdphi,res},
res=Table[
usdphi=Table[phi[[1+(j*m+p)*m;;m+(j*m+p)*m]],{j,0,per-1}];
Total[Abs[Fourier[#]]^2&/@usdphi],
{p,0,m-1}
];
Throw[Transpose[res]]
]];
gaus[x_,{x0_,p0_}]:=Exp[-(x-x0)^2/(4.*(2.\[Pi]/m)^2)+(I(x-x0)*p0 )/hbar];
gausIni[point_]:=#/Norm[#]&[q2p[gaus[#,point]&/@qSpec]];
planckIni[point_]:=Block[
{X=Floor[(m*point[[1]])/(2.\[Pi])],P=Floor[(m*point[[2]])/(2.\[Pi])],nonzero},
nonzero=Table[1./Sqrt[m] Exp[-I((n-1)+m*P)*X*2.\[Pi]/m],{n,1,m}];
Return[ArrayPad[nonzero,{m*P-1,dim-(m*P-1)-m}]]
];

q2p[lis_]:=InverseFourier[lis];
p2q[lis_]:=Fourier[lis];
evo[phi_]:=q2p[qPhase*p2q[phi]]*pPhase;
evoPath[ini_]:=Catch[
Block[{path={ini}},
Do[
AppendTo[path,evo[path[[-1]]]];
,{trajLen}
];
Throw[path]
]
];
(*Phase Space Representation*)
(*Return a 2-D list, p\[LeftDoubleBracket]x,p\[RightDoubleBracket]*)
phi2Ph[phi_]:=Catch[
Block[
{usdphi,res},
res=Table[
usdphi=Table[phi[[1+(j*m+p)*m;;m+(j*m+p)*m]],{j,0,per-1}];
Total[Abs[Fourier[#]]^2&/@usdphi],
{p,0,m-1}
];
Throw[Transpose[res]]
]];
obj2Ph[obj_]:=Position[xpGrid,SortBy[xpGrid,dist[#,obj]&][[1]]][[1]][[1]];
