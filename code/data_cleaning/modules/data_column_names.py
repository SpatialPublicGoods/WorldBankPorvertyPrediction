
class columname:
    

    def __init__(self):

        self.year = "Year"
        self.region = "Region"
        self.provincia = "Provincia"
        self.distrito = "Distrito"
        self.ubigeo = "Ubigeo"
        
        self.PIP = self.PIP()
        self.SIAF = self.SIAF()
        self.infogob = self.infogob()
        self.Onpe = self.Onpe()
        self.District = self.District()


    class District:

        def __init__(self):

            self.region = 'nombdep'
            self.provincia = 'nombprov'
            self.distrito = 'nombdist'
            self.ubigeo = 'ubigeo'

    class Onpe:

        def __init__(self):
        
            self.Spending = self.Spending()
            self.Results = self.Results()

        class Spending:

            def __init__(self):

                self.region = "DEPARTAMENTO"	
                self.province = "PROVINCIA"	
                self.district = "DISTRITO"
                self.type_political_organization = "TIPO DE OP"	
                self.name_political_organization = "ORGANIZACIÓN POLÍTICA (OP)"	
                self.dni = "DNI"	
                self.candidate_name = "CANDIDATO"	
                self.sex = "GÉNERO"	
                self.age = "EDAD"	
                self.office_position = "CARGO"	
                self.winner = "AUTORIDAD ELECTA"	
                self.delivery_type = "ENTREGA"	
                self.delivery_on_time = "ESTADO"	
                self.delivery_date = "FECHA DE PRESENTACIÓN"	
                self.income = "INGRESOS"	
                self.expenditure = "GASTOS"

        class Results:

            def __init__(self):

                self.ubigeo = "ubigeo"
                self.region = "departamento"
                self.provincia = "provincia"
                self.distrito = "distrito"
                self.election_type = "tipo_eleccion"
                self.acta = "mesa"
                self.acta_status = "estado_mesa"
                self.ballot_place = "ubicacion_en_cedula"
                self.type_political_organization = "tipo_agrupacion"
                self.political_organization_code = "codigo_agrupacion"
                self.political_group = "agrupacion_politica"
                self.votes_obtained = "votos_obtenidos"
                self.total_voters = "electores_habiles"
                self.white_votes = "votos_blancos"
                self.null_votes = "votos_nulos"
                self.canceled_votes = "votos_impug"

                self.place_name = "local de votación"
                self.address = "dirección local de votación"
                

    class infogob:

        def __init__(self):

            self.region = "Region"
            self.provincia = "Provincia"
            self.distrito = "Distrito"
            self.ubigeo = "Ubigeo"
            self.year_proceso = "Year"
            self.proceso = "Proceso"

            self.Resultados = self.Resultados()
            self.Padron = self.Padron()
            self.Candidatos = self.Candidatos()
            self.Autoridades = self.Autoridades()


        class Resultados:

            def __init__(self):

                self.numero_electores = 'Electores'
                self.porcentaje_participacion = '% Participacion'
                self.votos_emitidos = 'Votos emitidos'
                self.votos_validos = 'Votos validos'
                self.org_politica = 'Organizacion Politica'
                self.tipo_org_politica = 'Tipo Organizacion Politica'
                self.votos = 'Votos'
                self.porcentaje_votos = '% Votos'

        class Padron:

            def __init__(self):

                self.numero_electores = 'Numero de electores'
                self.electores_varones = 'Electores varones'
                self.porcentaje_varones = '% Electores varones'
                self.electores_mujeres = 'Electores mujeres'
                self.porcentaje_mujeres = '% Electores mujeres'
                self.electores_jovenes = 'Electores jovenes'
                self.porcentaje_jovenes = '%Electores jovenes'
                self.electores_mayores = 'Electores mayores de 70 años'
                self.porcentaje_mayores = '% Electores mayores de 70 años'

        class Candidatos:

            def __init__(self):

                self.org_politica = 'Organizacion Politica'
                self.tipo_org_politica = 'Tipo Organizacion Política'
                self.cargo = 'Cargo'
                self.numero = 'N°'
                self.primer_apellido = 'Primer apellido'
                self.segundo_apellido = 'Segundo apellido'
                self.prenombres = 'Prenombres'
                self.sexo = 'Sexo'
                self.joven = 'Joven'
                self.nativo = 'Nativo'

        class Autoridades:
            
            def __init__(self):

                self.cargo = 'Cargo electo'
                self.primer_apellido = 'Primer apellido'
                self.segundo_appelido = 'Segundo apellido'
                self.prenombres = 'Prenombres'
                self.org_politica = 'Organizacion Politica'
                self.tipo_org_politica = 'Tipo Organizacion Politica'
                self.sexo = 'Sexo'
                self.joven = 'Joven'
                self.nativo = 'Nativo'
                self.votos_org_politica = 'Votos organizacion politica'
                self.porcentaje_org_politica = '% Votos organizacion politica'


    class PIP:

        def __init__(self):

            self.year = 'ANO_PROCESADO'
            self.month = 'MES_PROCESADO'
            self.day = 'DIA_PROCESADO'

            self.year_viable = 'ANO_VIABLE'
            self.month_viable = 'MES_VIABLE'
            self.day_viable = 'DIA_VIABLE'

            self.finish_date = 'FEC_CIERRE'
            self.registry_date = 'FECHA_REGISTRO'
            self.viable_date = 'FECHA_VIABILIDAD'
            self.time_length = 'project_lenght'
            self.approval_time_length = 'project_approval_lenght'

            self.funcion = 'FUNCION'
            self.programa = 'PROGRAMA'
            self.subprograma = 'SUBPROGRAMA'

            self.gov_level = 'NIVEL'
            self.sector = 'SECTOR'
            self.project_id = 'CODIGO_UNICO'
            self.snip_id = 'CODIGO_SNIP'
            self.project_name = 'NOMBRE_INVERSION'
            self.entity = 'ENTIDAD'
            self.OPMI = 'UNIDAD_OPMI_EJE'
            self.UF = 'UNIDAD_UF_EJE'
            self.UEI = 'UNIDAD_UEI_EJE'
            self.executer = 'DESC_EJECUTORA_EJECUCION'
            self.amount_viable = 'MONTO_VIABLE'
            self.amount_updated = 'MONTO_ACTUALIZADO'
            self.status = 'ESTADO'
            self.situation = 'SITUACION'

            self.pmi_year1 = 'PMI_ANIO_1'
            self.pmi_year2 = 'PMI_ANIO_2'
            self.pmi_year3 = 'PMI_ANIO_3'
            self.pmi_year4 = 'PMI_ANIO_4'
            self.last_disburse = 'ULT_DEV'
            
            self.form_closure = 'INFORME_CIERRE'
            self.cum_disbursed_last_year = 'DEVEN_ACUMUL_ANIO_ANT'
            self.cum_disbursed_current_year = 'DEVEN_ACTUAL_ANIO_ACT'
            self.pia_updated = 'PIA_ANIO_ACT'
            # self.day = 'PIM_ANIO_ACT'
            # self.day = 'CERTIFIC_ANIO_ACT'
            # self.day = 'COMPROM_ANUAL_ANIO_ACT'
            self.amount_balance = 'SALDO_EJECUTAR'
            self.project_stage = 'ETAPA'

            self.amount_total_value = 'MTO_VALORIZACION'
            self.ubigeo = 'UBIGEO'
            self.region = 'DEPARTAMENTO'
            self.provincia = 'PROVINCIA'
            self.distrito = 'DISTRITO'
            self.lat = 'LATITUD'
            self.lon = 'LONGITUD'

            self.lat_project = 'lat_project'
            self.lon_project = 'lon_project'

            self.gov_level_siaf = 'NIVEL_SIAF'

            self.regime = 'MARCO'
            self.format_type = 'DES_TIPO_FORMATO'
            self.format_modality = 'DES_MODALIDAD'
            self.registered_pmi = 'REGISTRADO_PMI'

            # NEW VARIABLES:
            self.jurisdiction = "JURISDICTION_LEVEL"

            # self.day = 'DEV_01'
            # self.day = 'DEV_02'
            # self.day = 'DEV_03'
            # self.day = 'DEV_04'
            # self.day = 'DEV_05'
            # self.day = 'DEV_06'
            # self.day = 'DEV_07'
            # self.day = 'DEV_08'
            # self.day = 'DEV_09'
            # self.day = 'DEV_10'
            # self.day = 'DEV_11'
            # self.day = 'DEV_12'

            # self.function = 'FUNCION'
            # self.program = 'PROGRAMA'
            # self.subprogram = 'SUBPROGRAM'
            # self.day = 'TIENE_F12B'
            # self.day = 'FECHA_ULT_ACT_F12B'
            # self.day = 'ULT_PERIODO_REG_F12B'
            # self.day = 'TIENE_AVAN_FIS'
            self.progress_build = 'AVAN_FISICO_F12B'
            self.progress_spend = 'AVAN_EJECUC'
            # self.day = 'FEC_ULT_AVAN_FIS'
            # self.day = 'DES_TIPOLOGIA'
            # self.day = 'PROG_ACTUAL_ANIO_ACT'
            # self.day = 'EXPEDIENTE_TECNICO'
            # self.day = 'SANEAMIENTO'
            # self.day = 'PRIOR_GN'
            # self.day = 'PRIOR_GR'
            # self.day = 'PRIOR_GL'
            # self.day = 'SECTOR_SIAF'
            # self.day = 'PLIEGO_SIAF'
            # self.day = 'EJECUTORA_SIAF'
            # self.day = 'ANIO_PROCESO'

    class SIAF:

        def __init__(self):

            self.year = 'ANO_EJE' 
            self.month = 'MES_EJE' 
            self.gov_level = 'TIPO_GOBIERNO' 
            self.gov_level_name = 'TIPO_GOBIERNO_NOMBRE' 
            self.sector = 'SECTOR'
            self.sector_name = 'SECTOR_NOMBRE' 
            self.pliego = 'PLIEGO' 
            self.pliego_name = 'PLIEGO_NOMBRE' 
            
            self.sector_ejecutora = 'SEC_EJEC' 
            self.ejecutora = 'EJECUTORA'
            self.ejecutora_name = 'EJECUTORA_NOMBRE' 
            self.departamento_ejecutora = 'DEPARTAMENTO_EJECUTORA'
            self.departamento_ejecutora_nombre = 'DEPARTAMENTO_EJECUTORA_NOMBRE' 
            self.provincia_ejecutora = 'PROVINCIA_EJECUTORA'
            self.provincia_ejecutora_nombre = 'PROVINCIA_EJECUTORA_NOMBRE' 
            self.distrito_ejecutora = 'DISTRITO_EJECUTORA'
            self.distrito_ejecutora_nombre = 'DISTRITO_EJECUTORA_NOMBRE' 
            
            self.budget_source = 'FUENTE_FINANC'
            self.budget_source_name = 'FUENTE_FINANC_NOMBRE' 
            
            self.specific_expenditure = 'ESPECIFICA_NOMBRE' 
            self.specific_expenditure_det = 'ESPECIFICA_DET'
            self.specific_expenditure_det_name = 'ESPECIFICA_DET_NOMBRE' 

            self.item = 'RUBRO' 
            self.item_name = 'RUBRO_NOMBRE' 
            self.resource_type = 'TIPO_RECURSO'
            self.resource_type_name = 'TIPO_RECURSO_NOMBRE' 
            self.category = 'CATEG_GASTO' 
            self.category_name = 'CATEG_GASTO_NOMBRE'

            self.generica = 'GENERICA' 
            self.generica_name = 'GENERICA_NOMBRE' 
            self.subgenerica = 'SUBGENERICA'
            self.subgenerica_name = 'SUBGENERICA_NOMBRE' 
            self.subgenerica_det = 'SUBGENERICA_DET' 
            self.subgenerica_det_name = 'SUBGENERICA_DET_NOMBRE'

            self.amount_pia = 'MONTO_PIA' 
            self.amount_pim = 'MONTO_PIM' 
            self.amount_certified = 'MONTO_CERTIFICADO'
            self.amount_promised_yearly = 'MONTO_COMPROMETIDO_ANUAL' 
            self.amount_promised = 'MONTO_COMPROMETIDO' 
            self.amount_signed = 'MONTO_DEVENGADO'
            self.amount_disbursed = 'MONTO_GIRADO'

            # Project identifiers:
            self.budget_program = 'PROGRAMA_PPTO'
            self.budget_program_name = 'PROGRAMA_PPTO_NOMBRE' 
            self.project_id = 'PRODUCTO_PROYECTO'
            self.project_name = 'PRODUCTO_PROYECTO_NOMBRE' 
            self.activity_id = 'ACTIVIDAD_ACCION_OBRA'
            self.activity_name = 'ACTIVIDAD_ACCION_OBRA_NOMBRE' 

            self.function = 'FUNCION' 
            self.function_name = 'FUNCION_NOMBRE'
            self.functional_division = 'DIVISION_FUNCIONAL' 
            self.functional_division_name = 'DIVISION_FUNCIONAL_NOMBRE' 
            self.functional_group = 'GRUPO_FUNCIONAL'
            self.functional_group_name = 'GRUPO_FUNCIONAL_NOMBRE' 

            # Other potential identifiers:
            self.transaction_type = 'TIPO_TRANSACCION'
            self.function = 'SEC_FUNC' 
            self.project_type = 'TIPO_ACT_PROY' 
            self.goal = 'META' 
            self.purpose = 'FINALIDAD' 
            self.goal_name = 'META_NOMBRE'
            self.region_goal = 'DEPARTAMENTO_META'
            self.region_goal_name = 'DEPARTAMENTO_META_NOMBRE' 

            # New variables:
            self.n_projects = "n_projects"


            # Variable Names:
            self.Generica = self.Generica()


        class Generica:

            def __init__(self):
                
                self.capital_nofin = 'ADQUISICION DE ACTIVOS NO FINANCIEROS'
                self.goods_services = 'BIENES Y SERVICIOS'
                self.transfers = 'DONACIONES Y TRANSFERENCIAS'
                self.other ='OTROS GASTOS'
                self.pension ='PENSIONES Y OTRAS PRESTACIONES SOCIALES'
                self.social_security ='PERSONAL Y OBLIGACIONES SOCIALES' 
                self.public_debt ='SERVICIO DE LA DEUDA PUBLICA'
                self.capital_fin = 'ADQUISICION DE ACTIVOS FINANCIEROS'



        class SubGenerica:

            def __init__(self):
                
                self.capital_nofin = 'ADQUISICION DE ACTIVOS NO FINANCIEROS'
                self.goods_services = 'BIENES Y SERVICIOS'
                self.transfers = 'DONACIONES Y TRANSFERENCIAS'
                self.other ='OTROS GASTOS'
                self.pension ='PENSIONES Y OTRAS PRESTACIONES SOCIALES'
                self.social_security ='PERSONAL Y OBLIGACIONES SOCIALES' 
                self.public_debt ='SERVICIO DE LA DEUDA PUBLICA'
                self.capital_fin = 'ADQUISICION DE ACTIVOS FINANCIEROS'


