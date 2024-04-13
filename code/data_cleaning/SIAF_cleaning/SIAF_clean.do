* SIAF Data cleaning

* categories
/*
 funcion_nombre 
 categoria_gasto_nombre

*/

*  departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre ano_eje mes_eje nivel_gobierno nivel_gobierno_nombre

/*
forvalues y=2010(1)2021 {
	 
	
	import delimited   departamento_ejecutora_nombre  provincia_ejecutora_nombre  distrito_ejecutora_nombre ano_eje mes_eje    monto_pia monto_pim monto_comprometido monto_devengado monto_girado using  "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\SIAF\\`y'-Gasto.csv" ,   clear rowrange(1:500)
	
	*keep departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre ano_eje mes_eje nivel_gobierno nivel_gobierno_nombre   funcion_nombre  categoria_gasto_nombre
	
	compress
	
	export delimited  using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\gastos`y'.csv", replace 

	
}


*/
/*
forvalues y=2010(1)2019 {
	 
	
	import delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\\`y'Gasto.csv", clear 

	** collapse with negative
	preserve
	collapse (sum)  monto_pia monto_pim monto_comprometido monto_devengado monto_girado, by( departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre ano_eje mes_eje nivel_gobierno nivel_gobierno_nombre funcion_nombre categoria_gasto_nombre)
	tempfile wneg
	save `wneg', replace
	restore
	
	** collapse without negatives
	foreach var in  monto_pia monto_pim monto_comprometido monto_devengado monto_girado {
		replace `var'=0 if `var'<0
		
	}
	
	collapse (sum)  monto_pia monto_pim monto_comprometido monto_devengado monto_girado, by( departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre ano_eje mes_eje nivel_gobierno nivel_gobierno_nombre funcion_nombre categoria_gasto_nombre)
	
	renvars   monto_pia monto_pim monto_comprometido monto_devengado monto_girado, postfix(_noneg)
	merge 1:1 departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre ano_eje mes_eje nivel_gobierno nivel_gobierno_nombre funcion_nombre categoria_gasto_nombre using `wneg'
	drop _merge
 drop if distrito_ejecutora_nombre==" "	
 
 save "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\\`y'Gasto_collapsed", replace
 
 
}
*/

**** Append all years

/*
drop _all
forvalues y=2010(1)2019 {

append using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\\`y'Gasto_collapsed", force
}


	export delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Gasto_allyears.csv", replace 
*/







/***** INGRESOS 

forvalues y=2010(1)2019 {
	 
	
	import delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\SIAF\\`y'-Ingreso.csv", clear 

	** collapse with negative
	preserve
	collapse (sum) monto_pia monto_pim monto_recaudado , by( ano_doc mes_doc nivel_gobierno_nombre nivel_gobierno departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre rubro rubro_nombre)
	tempfile wneg
	save `wneg', replace
	restore
	
	** collapse without negatives
	foreach var in  monto_pia monto_pim monto_recaudado {
		replace `var'=0 if `var'<0
		
	}
	
	collapse (sum)  monto_pia monto_pim monto_recaudado, by( ano_doc mes_doc nivel_gobierno_nombre nivel_gobierno departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre rubro rubro_nombre)
	
	renvars  monto_pia monto_pim monto_recaudado, postfix(_noneg)
	merge 1:1 ano_doc mes_doc nivel_gobierno_nombre nivel_gobierno departamento_ejecutora departamento_ejecutora_nombre provincia_ejecutora provincia_ejecutora_nombre distrito_ejecutora distrito_ejecutora_nombre rubro rubro_nombre using `wneg'
	drop _merge
 drop if distrito_ejecutora_nombre==" "	
 
 save "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\\`y'Ingreso_collapsed", replace
 
 
}
*/

** Append
/*
drop _all
forvalues y=2010(1)2019 {

append using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\\`y'Ingreso_collapsed", force
}


	export delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Ingreso_allyears.csv", replace 
	
	
	*/
	
*** Merge with UBIGEO
** INGRESO
/*

import delimited "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\Housing\TB_UBIGEOS (1).csv", clear 
rename distrito distrito_ejecutora_nombre
rename departamento departamento_ejecutora_nombre
rename provincia provincia_ejecutora_nombre
				  
tempfile ubigeos
save `ubigeos', replace


import delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Ingreso_allyears.csv", clear



foreach var in departamento_ejecutora provincia_ejecutora distrito_ejecutora {
	
	egen aux=max(`var'), by(departamento_ejecutora_nombre provincia_ejecutora_nombre distrito_ejecutora_nombre)
	replace `var'=aux if `var'==.
	drop aux
	
}




tostring departamento_ejecutora provincia_ejecutora distrito_ejecutora, replace


foreach var in departamento_ejecutora provincia_ejecutora distrito_ejecutora {

replace `var'="0"+`var' if `var'=="1" |`var'=="2"|`var'=="3" |`var'=="4" |`var'=="5" |`var'=="6" |`var'=="7" |`var'=="8" |`var'=="9"
}

gen ubigeo_inei=departamento_ejecutora+provincia_ejecutora+distrito_ejecutora

replace ubigeo="30610"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="EL PORVENIR"
replace ubigeo="30611"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="LOS CHANKAS"
replace ubigeo="30609"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="ROCCHACC"
replace ubigeo="50412"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="HUANTA" & distrito_ejecutora_nombre=="CHACA"
replace ubigeo="50411"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="HUANTA" & distrito_ejecutora_nombre=="PUCACOLPA"
replace ubigeo="50511"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="LA MAR" & distrito_ejecutora_nombre=="ORONCCOY"
replace ubigeo="80914"	 if departamento_ejecutora_nombre=="CUSCO" & provincia_ejecutora_nombre=="LA CONVENCION" & distrito_ejecutora_nombre=="MEGANTONI"
replace ubigeo="80913"	 if departamento_ejecutora_nombre=="CUSCO" & provincia_ejecutora_nombre=="LA CONVENCION" & distrito_ejecutora_nombre=="VILLA KINTIARINA"
replace ubigeo="90722"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="PICHOS"
replace ubigeo="90721"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="ROBLE"
replace ubigeo="90723"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="SANTIAGO DE TUCUMA"
replace ubigeo="100113	"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="HUANUCO"	 & distrito_ejecutora_nombre=="SAN PABLO DE PILLAO"
replace ubigeo="100608"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="CASTILLO GRANDE"
replace ubigeo="100607"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="PUCAYACU"
replace ubigeo="100609"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="PUEBLO NUEVO"
replace ubigeo="100610"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="SANTO DOMINGO DE ANDA"
replace ubigeo="100704"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="MARANON" & distrito_ejecutora_nombre=="LA MORADA"
replace ubigeo="100705"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="MARANON" & distrito_ejecutora_nombre=="SANTA ROSA DE ALTO YANAJANCA"
replace ubigeo="120609"	 if departamento_ejecutora_nombre=="JUNIN" & provincia_ejecutora_nombre=="SATIPO" & distrito_ejecutora_nombre=="VIZCATAN DEL ENE"
replace ubigeo="211105"	 if departamento_ejecutora_nombre=="PUNO" & provincia_ejecutora_nombre=="SAN ROMAN" & distrito_ejecutora_nombre=="SAN MIGUEL"
replace ubigeo="230111"	 if departamento_ejecutora_nombre=="TACNA" & provincia_ejecutora_nombre=="TACNA" & distrito_ejecutora_nombre=="LA YARADA-LOS PALOS"




destring ubigeo_inei, replace 
merge m:1 ubigeo_inei using `ubigeos' 
drop _merge

	export delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Ingreso_allyears_ubigeos.csv", replace 
*/


* GASTO



import delimited "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\Housing\TB_UBIGEOS (1).csv", clear 
rename distrito distrito_ejecutora_nombre
rename departamento departamento_ejecutora_nombre
rename provincia provincia_ejecutora_nombre
				  
tempfile ubigeos
save `ubigeos', replace


import delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Gasto_allyears.csv", clear



foreach var in departamento_ejecutora provincia_ejecutora distrito_ejecutora {
	
	egen aux=max(`var'), by(departamento_ejecutora_nombre provincia_ejecutora_nombre distrito_ejecutora_nombre)
	replace `var'=aux if `var'==.
	drop aux
	
}




tostring departamento_ejecutora provincia_ejecutora distrito_ejecutora, replace


foreach var in departamento_ejecutora provincia_ejecutora distrito_ejecutora {

replace `var'="0"+`var' if `var'=="1" |`var'=="2"|`var'=="3" |`var'=="4" |`var'=="5" |`var'=="6" |`var'=="7" |`var'=="8" |`var'=="9"
}

gen ubigeo_inei=departamento_ejecutora+provincia_ejecutora+distrito_ejecutora

replace ubigeo="30610"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="EL PORVENIR"
replace ubigeo="30611"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="LOS CHANKAS"
replace ubigeo="30609"	 if departamento_ejecutora_nombre=="APURIMAC" & provincia_ejecutora_nombre=="CHINCHEROS" & distrito_ejecutora_nombre=="ROCCHACC"
replace ubigeo="50412"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="HUANTA" & distrito_ejecutora_nombre=="CHACA"
replace ubigeo="50411"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="HUANTA" & distrito_ejecutora_nombre=="PUCACOLPA"
replace ubigeo="50511"	 if departamento_ejecutora_nombre=="AYACUCHO" & provincia_ejecutora_nombre=="LA MAR" & distrito_ejecutora_nombre=="ORONCCOY"
replace ubigeo="80914"	 if departamento_ejecutora_nombre=="CUSCO" & provincia_ejecutora_nombre=="LA CONVENCION" & distrito_ejecutora_nombre=="MEGANTONI"
replace ubigeo="80913"	 if departamento_ejecutora_nombre=="CUSCO" & provincia_ejecutora_nombre=="LA CONVENCION" & distrito_ejecutora_nombre=="VILLA KINTIARINA"
replace ubigeo="90722"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="PICHOS"
replace ubigeo="90721"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="ROBLE"
replace ubigeo="90723"	 if departamento_ejecutora_nombre=="HUANCAVELICA" & provincia_ejecutora_nombre=="TAYACAJA" & distrito_ejecutora_nombre=="SANTIAGO DE TUCUMA"
replace ubigeo="100113	"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="HUANUCO"	 & distrito_ejecutora_nombre=="SAN PABLO DE PILLAO"
replace ubigeo="100608"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="CASTILLO GRANDE"
replace ubigeo="100607"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="PUCAYACU"
replace ubigeo="100609"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="PUEBLO NUEVO"
replace ubigeo="100610"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="LEONCIO PRADO" & distrito_ejecutora_nombre=="SANTO DOMINGO DE ANDA"
replace ubigeo="100704"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="MARANON" & distrito_ejecutora_nombre=="LA MORADA"
replace ubigeo="100705"	 if departamento_ejecutora_nombre=="HUANUCO" & provincia_ejecutora_nombre=="MARANON" & distrito_ejecutora_nombre=="SANTA ROSA DE ALTO YANAJANCA"
replace ubigeo="120609"	 if departamento_ejecutora_nombre=="JUNIN" & provincia_ejecutora_nombre=="SATIPO" & distrito_ejecutora_nombre=="VIZCATAN DEL ENE"
replace ubigeo="211105"	 if departamento_ejecutora_nombre=="PUNO" & provincia_ejecutora_nombre=="SAN ROMAN" & distrito_ejecutora_nombre=="SAN MIGUEL"
replace ubigeo="230111"	 if departamento_ejecutora_nombre=="TACNA" & provincia_ejecutora_nombre=="TACNA" & distrito_ejecutora_nombre=="LA YARADA-LOS PALOS"




destring ubigeo_inei, replace 
merge m:1 ubigeo_inei using `ubigeos' 
drop _merge

	export delimited   using "C:\Users\wb277714\OneDrive - WBG\Research\Monitoring\peru\big_data\admin\SIAF\Gasto_allyears_ubigeos.csv", replace 
*/

