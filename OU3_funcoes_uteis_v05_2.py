# Coleção de funções para a disciplina de OU3
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def f_Pvap_Antoine_db(Temp, i_comp, dados):
    #import numpy as np
    ''' Função que calcula a pressão de vapor, segundo a equação de Antoine, para o componente
      i_comp presente no databank_properties.pickle.
      Equação de Antoine: Pvap = exp(A - B /(Temp + C)), com: 
      [Temp] = K
      [Pvap] = mmHg
      Entrada (argumentos da função)
      Temp   = temperatura em K para a qual será calculada a Pvap
      i_comp = inteiro que corresponde ao número do componente no banco de dados
               campo/column/key = 'num'
      dados  = pandas dataframe com os dados lidos do arquivo
      Saida: tupla
      Pvap - pressão de vapor do i_comp em mmHg
      par = dicionário com os parâmetros A, B e C da equação de Antoine
    '''
    # param <- as.numeric(param)
    par_array = np.array(dados[dados['num'] == i_comp][['pvap_a','pvap_b','pvap_c']])[0]
    par = {'a': par_array[0], 'b': par_array[1], 'c': par_array[2]}
    a = par['a']
    b = par['b']
    c = par['c']
    Pvap = np.exp(a - b/(Temp + c))
    # attr(x = Pvap, which = "units") <- "mmHg"
    return Pvap, par

def f_K_Raoult_db(T_eq, P_eq, lista_componentes, dados):
    # import numpy as np
    ''' Função para o cálculo da volatilidade segundo a Lei de Raoult:
        - fase vapor -> mistura de gás ideal
        - fase líquida -> solução normal
        K = P_vap(Teq) / P_eq
        Entrada (argumentos da função)
        T_eq - temperatura de equilíbrio em K
        P_eq - pressão de equilíbrio em mmHg
        lista_componentes - lista com os números inteiro dos componentes no databank
        dados - pandas dataframe com os dados do databank_properties.pickle
        Saida: tupla
        K_comp - np.array com os valores da volatilidade na ordem da lista_componentes
        P_vap_comp - np.array com os valores de P_vap segundo a equação de Antoine e os parâmetros
                    do databank_properties.pickle
    '''
    nc = len(lista_componentes)
    P_vap_comp = np.empty(nc)
    K_comp = np.empty(nc)
    k = 0
    for i_comp in lista_componentes:
        P_vap_comp[k], par = f_Pvap_Antoine_db(T_eq, i_comp, dados)
        K_comp[k] = P_vap_comp[k] / P_eq
        k += 1
    return K_comp, P_vap_comp

def f_Pb_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de bolha
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  x = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_T(Temp,P,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Temperatura do ponto de orvalho
      Entrada:
      Temp - temperaura de equilíbrio em K - variável implícita da equação
      P - pressão de equilíbrio em mmHg
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(Temp) == float):
    Temp = np.array([Temp])
  nc = len(z)
  nr = len(Temp)
  MP = np.empty((nr,nc))
  y = z
  for i, T_vez in enumerate(Temp):
    K_comp = f_K_Raoult_db(T_vez, P, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Pb_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de bolha
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  x = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = K_comp * x
  f = 1 - np.sum(MP, axis=1)
  return f

def f_Po_P(P,Temp,z,lista_componentes,dados):
  ''' Função que retorna o resíduo para o cálculo da Pressão do ponto de orvalho
      Entrada:
      P - pressão de equilíbrio em mmHg - variável implícita da equação
      Temp - temperaura de equilíbrio em K 
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saida:
      f - resíduo da função (f = 0 -> solução)
  '''
  if (type(P) == float):
    P = np.array([P])
  nc = len(z)
  nr = len(P)
  MP = np.empty((nr,nc))
  y = z
  for i, P_vez in enumerate(P):
    K_comp = f_K_Raoult_db(Temp, P_vez, lista_componentes, dados)[0]
    MP[i,:] = y / K_comp
  f = 1 - np.sum(MP, axis=1)
  return f

def f_calculo_PbPo_db(vp, x_pot, z, lista_componentes, dados):
    ''' Função para o cálculo das temperatura ou pressões do ponto de bolha 
          e do ponto de orvalho ( [T] em K e [P] em mmHg)
        Entradas:
        vp - variável do problema 'T' ou 'P' - string
        x_pot - valor de pressão ou temperatura dado
        z - composição da carga em fração molar
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas:
        Se vp == 'T' -> T_Pb, T_Po, T_eb_comp = lista com as temperaturas de
                        ebulição normal dos componentes
        Se vp == 'P' -> P_Pb, P_Po, M_P_vap = matriz com as pressões de 
                        vapor dos componentes nas T_eb_comp
    '''
    #from scipy.optimize import fsolve
    nc = len(lista_componentes)
    T_eb_comp = np.zeros((nc,))
    i = 0
    for i_comp in lista_componentes:
        T_eb_comp[i] = float(dados[dados['num'] == i_comp]['boiling_point'])
        i += 1
    if (vp == 'T'):
        P_eq = x_pot
        #T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        #T_eb_comp = T_eb_comp.tolist()
        T_guest = (min(T_eb_comp) + max(T_eb_comp) )/2
        T_Pb = fsolve(f_Pb_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        T_Po = fsolve(f_Po_T, T_guest, args=(P_eq, z, lista_componentes, dados))[0]
        return (T_Pb, T_Po, T_eb_comp)
    if (vp == 'P'):
        T_eq = x_pot
        #T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
        #T_eb_comp = T_eb_comp.tolist()
        P_vap_eb_comp = np.empty(nc)
        k = 0
        for i_comp in lista_componentes:
          P_vap_eb_comp[k] = f_Pvap_Antoine_db(T_eq, i_comp, dados)[0]
          k += 1
        P_guest = (np.min(P_vap_eb_comp) + np.max(P_vap_eb_comp))/2
        P_Pb = fsolve(f_Pb_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        P_Po = fsolve(f_Po_P, P_guest, args=(T_eq, z, lista_componentes, dados))[0]
        return (P_Pb, P_Po, P_vap_eb_comp)

def f_res_RR_flash_db(fv, z, P, Temp, lista_componentes, dados):
    ''' Função que determina o resíduo da equação de Rachford-Rice para o flash
        multicomponente na solução para encontrar fv (fração vaporizada da carga)
      Entrada:
      fv - fração vaporizada da carga - variável implícita
      z - composição da carga em fração molar
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas:
      res - resíduo na busca da solução - res = 0 -> solução
    '''
    nc = len(lista_componentes)
    if (type(fv) == float):
      fv = np.array([fv])
    nr = len(fv)
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    M_parc = np.empty((nr, nc))
    num = z * K_comp
    for i, fv_vez in enumerate(fv):
        den = 1.0 + fv_vez*(K_comp - 1.0)
        M_parc[i,:] = num / den
    res = 1.0 - np.sum(M_parc, axis=1)
    return res

def f_sol_RR_flash_db(z, P, Temp, lista_componentes, dados):
    ''' Função que resolve a equação de Rachford-Rice e encontra a fv 
        (fração vaporizada da carga)
        Entrada:
        z - composição da carga em fração molar
        P - pressão do flash em mmHg
        T - temperatura do flash em K
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas: {dicionário}
        fv_flash - fração vaporizada - solução do flash
        x_eq - composição do líquido no equilíbrio
        y_eq - composição do vapor no equilíbrio
        K_comp - volatilidade dos componentes
        alpha_comp - volatilidade relativa em relação ao componente chave pesado (i_chk)
    '''
    fv_guest = 0.5
    fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
    K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
    num = z * K_comp
    den = 1.0 + fv_flash*(K_comp - 1.0)
    y_eq = num / den
    x_eq = y_eq / K_comp
    i_chk = np.argmin(K_comp)
    alpha_comp = K_comp/K_comp[i_chk]
    return {'fv_flash': fv_flash, 'x_eq': x_eq, 'y_eq': y_eq, 'K_comp': K_comp,
            'alpha_comp':alpha_comp}

def f_sol_ELV_2c_db(Temp, P, lista_componentes, dados):
    ''' Função para o cálculo do ELV em um sistema binário ideal (Lei de Raoult)
      Entrada:
      P - pressão do flash em mmHg
      T - temperatura do flash em K
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas: tupla de vetores
      x_eq - concentrações do componentes no ELV na fase líquida
      y_eq - concentrações do componentes no ELV na fase vapor
    '''
    nc = len(lista_componentes)
    P_vap_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[1]
    v_rhs = np.array([1,1,0,0])
    A_elv = np.array([[1,1,0,0],
                      [0,0,1,1],
                      [P_vap_comp[0],0,-P,0],
                      [0, P_vap_comp[1], 0, -P]])
    x_sol = np.linalg.inv(A_elv) @ v_rhs
    x_eq = np.empty(nc)
    y_eq = np.empty(nc)
    x_eq[0] = x_sol[0]
    x_eq[1] = x_sol[1]
    y_eq[0] = x_sol[2]
    y_eq[1] = x_sol[3]
    return (x_eq, y_eq)

def f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados):
    ''' Função para gerar um pandas.dataframe com n_pontos instâncias de dados
          do ELV de um sistema binário ideal
        Entradas:
          P_eq: pressão de equilíbrio em mmHg
          n_pontos: número de instâncias geradas
          lista_componentes: lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
          dados: dataframe com os dados do databank
        Saida: pandas.dataframe
          dados_elv: com as seguintes series: 'T', 'x1' e 'y1'
    '''
    T_eb_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point']
    T_eb_comp = T_eb_comp.tolist()
    T_faixa = np.linspace(T_eb_comp[0], T_eb_comp[1], n_pontos)
    dados_elv = pd.DataFrame({'T': T_faixa})
    for i, T in enumerate(dados_elv['T']):
        x_eq, y_eq = f_sol_ELV_2c_db(T, P_eq, lista_componentes, dados)
        dados_elv.loc[i,'x1'] = x_eq[0]
        dados_elv.loc[i,'y1'] = y_eq[0]
    return dados_elv

def f_reta_flash(x1, z1, fv):
  ''' Função da reta de operação do flash no diagrma y-x
      Entrada:
        x1: concentração do componente mais volátil na fase líquida
        z1: concentração do componente mais volátil na carga (F)
        fv: fração vaporizada
      Saida:
        y1 = composição do componente mais volátil na fase vapor pertencente
            a reta de operação de flash
  '''
  if (fv == 0):
    n_pontos = len(x1)
    y1 = np.linspace(z1,1.0,n_pontos)
  else:
    a = -(1.0 - fv)/ fv
    b = z1/fv
    y1 = a*x1 + b
  return y1

def f_gera_diag_y_x(P_eq, n_pontos, fig_name, fig_size, lista_componentes, dados):
    ''' Função que gera o diagrama y-x de uma mistura binária
        
        P_eq - pressão de equilíbrio em mmHg
    '''
    dados_elv = f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados)
    # Fazendo o gráfico
    fig, ax = plt.subplots(num=fig_name, figsize = fig_size)
    ax.plot(dados_elv['x1'], dados_elv['y1'], 'b', label='Equilíbrio')
    ax.plot(dados_elv['x1'], dados_elv['x1'], 'k', label= r'$y_1 = x_1$')
    # Adicionando texto nos eixos - descrição
    ax.set_xlabel('x1 - fase líquida')
    ax.set_ylabel('y1 - fase vapor')
    # Adicionando título para a figura
    ax.set_title('Diagrama y-x')
    # Adicionando um texto
    ax.text(0.4, 1.0, r'@$P_{eq} = 760.0 \, mmHg$')
    # Adicionando uma legenda
    ax.legend()
    # Adicionando linha vertical
    # ax1.vlines(z[0],T_Pb, T_Po, colors='m', linestyles='dashed')
    #ax1.axvline(sol_flash['x_eq'][0])
    #ax1.axvline(sol_flash['y_eq'][0])
    # Adicionando linha horizontal
    #ax1.hlines(T_flash,0, 1, colors='y', linestyles='solid')
    # Adicionando grade
    ax.grid()
    # Ajustando os ticks
    from matplotlib.ticker import (MultipleLocator) #AutoMinorLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.tick_params(which='minor', length=4, color='r')
    ax.tick_params(which='major', length=6, color='k')
    #ax1.grid(True, which='minor')
    #plt.show()
    return fig, ax


def f_adi_reta_flash(fig_f, ax_f, z, T_flash, P_flash, lista_componentes, dados, fv=float('NaN')):
    ''' Adiciona uma reta de flash no diagrama y-x'''
    import math
    # Verificação e cálculo da fração vaporizada da carga --> fv
    sol_flash = {'fv_flash': 0.0,
                 'x_eq': 0.0,
                 'y_eq': 0.0}
    if (math.isnan(fv)):
        sol_flash = f_sol_RR_flash_jvat_db(z, P_flash, T_flash, lista_componentes, dados)
        fv     = sol_flash['fv_flash']
        x_eq_1 = sol_flash['x_eq'][0]
        y_eq_1 = sol_flash['y_eq'][0]
    # determinação do menor valor em x para a reta de operação do flash
    if (fv == 0.0):
        x_min_fv = z[0]
        dados_ps = f_gera_dados_diag_H_ps(P_flash, 20, lista_componentes, dados)
        modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
        resp_pinch = f_Pinch(z, fv, modelos_ps)
        y_eq_1 = resp_pinch['y_p']
        x_eq_1 = resp_pinch['x_p']
    elif (fv == 1.0):
        x_min_fv = 0.0
        dados_ps = f_gera_dados_diag_H_ps(P_flash, 20, lista_componentes, dados)
        modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
        resp_pinch = f_Pinch(z, fv, modelos_ps)
        y_eq_1 = resp_pinch['y_p']
        x_eq_1 = resp_pinch['x_p']
    else:
        if (fv > 0):
            x_min_fv =(fv-z[0])/(fv-1.0)
            if (x_min_fv < 0):
              x_min_fv = 0.0
            dados_ps = f_gera_dados_diag_H_ps(P_flash, 20, lista_componentes, dados)
            modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
            resp_pinch = f_Pinch(z, fv, modelos_ps)
            y_eq_1 = resp_pinch['y_p']
            x_eq_1 = resp_pinch['x_p']
        if (fv < 0):
            x_min_fv = 0.0
            dados_ps = f_gera_dados_diag_H_ps(P_flash, 20, lista_componentes, dados)
            modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
            resp_pinch = f_Pinch(z, fv, modelos_ps)
            y_eq_1 = resp_pinch['y_p']
            x_eq_1 = resp_pinch['x_p']
    #x_eq_1 = sol_flash['x_eq'][0]
    #y_eq_1 = sol_flash['y_eq'][0]
    ax_f.axvline(x_eq_1, linestyle='dotted')
    ax_f.axhline(y_eq_1, linestyle='dotted')
    # Geração dos pontos da reta de operação do flash
    x1_graf = np.linspace(x_min_fv,z[0],20)
    y1_rf_graf = f_reta_flash(x1_graf,z[0], fv)
    # inserção da reta de flash do diagrama y-x
    ax_f.plot(x1_graf, y1_rf_graf, 'r', label= 'reta do flash')
    ax_f.vlines(z[0],0, z[0], colors='m', linestyles='dashed')
    ax_f.legend()
    # plt.show()
    return fig_f, ax_f


def f_cp_vap(Temp,param):
    '''Função que calcula cp do vapor como  gás ideal para os dados do 
        databank_properties.pickle
    '''
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    cp = a + b*Temp + c*Temp**2 + d*Temp**3
    # attr(x = cp, which = "units") <- "cal/mol_K"
    cp = 4.184 * cp # conversão de cal para J (Joules)
    #cp = 1000.0 * cp # conversão de mol para kmol
    return cp # J/mol/K


def f_cp_param(T1, lista_componentes, dados):
    '''Função que monta uma matriz com os parâmetros de todos os compoenente da 
        lista_componentes e também calcula o valor de cp de cada um deles na 
        temperatura T1
        Entradas:
        T1 = temperatura em K
        lista_componentes =
        dados =
        Saidas:
        v_cp = vetor com os valores de cp @T1 em cal/mol/K
        M_param = matriz com os quatro parâmetrso da equação do modelo de cp, sendo um
                  componente por linha na mesma ordem de lista_componentes
        '''
    nc = len(lista_componentes)
    M_param = np.empty((nc,4))
    v_cp = np.empty((nc))
    k = 0
    for i_num in lista_componentes:
        #print(i_num)
        param = dados [dados['num'] == i_num][['cp_a', 'cp_b', 'cp_c', 'cp_d']]
        param = param.to_numpy()[0]
        M_param[k,:] = param
        v_cp[k] = f_cp_vap(T1,param)
        k += 1
    # v_cp em J/mol/K
    return (v_cp, M_param)


def f_H_vap_ig_stream(y_stream, T_ref, Temp, lista_componentes, dados):
    '''Função para o cálculo para aentalpia de uma corrente na fase vapor e 
        considerada como gás ideal
        Entradas:
        y_stream = 
        T_ref = 
        Temp = 
        lista_componentes =
        dados =
        Saídas: em uma tupla
        H_stream = entalpia da corrente com composição y_stream
        DH = DeltaH dos componente de T_ref até Temp
    '''
    nc = len(lista_componentes)
    M_param = f_cp_param(T_ref, lista_componentes, dados)[1]
    DH = np.empty((nc,))
    for i in range(nc):
        DH[i] = integrate.quad(f_cp_vap, T_ref, Temp, args =(M_param[i,:],))[0]
    H_stream = y_stream @ DH
    # Entalpias em J/mol ou kJ/kmol
    return (H_stream, DH)


def f_DHvap_ClausiusClayperon_db(Temp, lista_componentes, dados):
    ''' Função para o cálculo da entalpia e vaporrização a partir de um modelo
        de pressão de vapor (modelo de Antoine)
        Entradas:
        Temp = temperatura na qual deseja-se o valor da entalpia de vaporização
        lista_componentes =
        dados =
        Saidas:
        DH_vap_comp_T = valor da entalpia de vaporização na temperatura Temp em K para
                        todos os componentes da lista_componentes em J/mol
    '''
    nc = len(lista_componentes)
    R = 1.987207 # cal/mol.K
    T1 = Temp - 10.0
    T2 = Temp + 10.0
    tt = (1/T1) - (1/T2)
    DH_vap_comp_T = np.zeros((nc,))
    k = 0
    for i_comp in lista_componentes:
        Pv1 = f_Pvap_Antoine_db(T1, i_comp, dados)[0]
        Pv2 = f_Pvap_Antoine_db(T2, i_comp, dados)[0]
        DH_vap_comp_T[k] = R * np.log(Pv2/Pv1) / tt
        k += 1
    return DH_vap_comp_T * 4.184


def f_DHvap_Watson_db(Temp, lista_componentes, dados):
    ''' Função para o cálculo da entalpia de vaporização em função da temperatura
            a partir da entalpia de vaporização medida no ponto de ebulição normal.
            Utiliza o modelo de Watson que corresponde a eq.4.13 da p. 100 do SVNA.
        Entradas:
          Temp = temperatura em K
          lista_componentes = 
          dados = 
        Saídas:
          DH_vap_comp = vetor com as entalpias em Temp em J/mol
    '''
    nc = len(lista_componentes)
    T_BP_comp = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    DH_vap_comp_bp = dados[dados['num'].isin(lista_componentes)]['delta_h_vap_bp'].to_numpy()
    Tc_comp = dados[dados['num'].isin(lista_componentes)]['critical_temp'].to_numpy()
    Tr_eb_comp = T_BP_comp / Tc_comp # Temperatura de ebulição reduzida
    DH_vap_comp_T = np.zeros((nc,))
    for i in range(nc):
        frac = (1.0 - (Temp/Tc_comp[i]))/(1.0 - (Tr_eb_comp[i]))
        DH_vap_comp_T[i] = DH_vap_comp_bp[i]*((frac)**(0.38))
    return DH_vap_comp_T * 4.184


def f_gera_mod_locais_ps(dados_ps):
    ''' Modelo locais do diagrama entalpia composição para usar no método
            Ponchon-Savarit & McCabe-Thiele
            Entrada:
            dados_elv = dataframe do pandas com os dados de:
                T = temperatura em K
                x1 = composição de equilíbrio na fase líquida (liquido saturado)
                y1 = composição de equilíbrio na fase vapor (vapor saturado)
                DH_vap_Watson = entalpia de vaporização da mistura, composição do vapor,
                                calculada com o modelo de Watson
                Hig_v = entalpia do vapor saturado (J/mol)
                Hig_l = entalpia do líquido saturado (J/mol)
            Saidas:
            modelos = lista com os objetos dos respecitovos modelos na ordem:
                      mod_H_x, mod_H_y, mod_y_x, mod_x_y
            r2_modelos = lista com os valores dos coeficientes de determinação dos
                         modelos estimados
    '''
    # Modelo local para Hig_l = f(x1)
    x_regr = dados_ps['x1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['Hig_l'].to_numpy()
    polinomio_2g = PolynomialFeatures(degree = 2)
    X_poli_2g = polinomio_2g.fit_transform(x_regr)
    mod_H_x = LinearRegression()
    mod_H_x.fit(X_poli_2g, y_regr)
    r2_H_x = r2_score(y_regr, mod_H_x.predict(X_poli_2g))
    # Modelo local para Hig_v = f(y1)
    x_regr = dados_ps['y1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['Hig_v'].to_numpy()
    polinomio_2g = PolynomialFeatures(degree = 2)
    X_poli_2g = polinomio_2g.fit_transform(x_regr)
    mod_H_y = LinearRegression()
    mod_H_y.fit(X_poli_2g, y_regr)
    r2_H_y = r2_score(y_regr, mod_H_y.predict(X_poli_2g))
    # Modelo local para y1 = f(x1) - 3º grau
    x_regr = dados_ps['x1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['y1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_y_x = LinearRegression()
    mod_y_x.fit(X_poli_3g, y_regr)
    r2_y_x = r2_score(y_regr, mod_y_x.predict(X_poli_3g))
    # Modelo local para x1 = f(y1) - 3º grau
    x_regr = dados_ps['y1'].to_numpy().reshape(-1,1)
    y_regr = dados_ps['x1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_x_y = LinearRegression()
    mod_x_y.fit(X_poli_3g, y_regr)
    r2_x_y = r2_score(y_regr, mod_x_y.predict(X_poli_3g))
    #
    r2_modelos = [r2_H_x, r2_H_y, r2_y_x, r2_x_y]
    modelos = [mod_H_x, mod_H_y, mod_y_x, mod_x_y]
    #
    return (modelos, r2_modelos)


def f_uso_mod_loc_ps(x_var_ind, nome_var_dep, modelos):
    '''Função para usar os modelos locais estimados com: f_gera_mod_locais_ps
        Entradas:
        x_var_ind = valor da variável independente para o ual deseja calcular
                    a variável dependente
        nome_var_dep = string com o nome da variável dependente, podendo ser os
                    seguintes: 'Hl', 'Hv', 'y1' e 'x1'
        modelos = listas dos modelos estimados anteriormente
        Saidas:
        resp = valor calculado para a variável dependente
    '''
    #
    polinomio_2g = PolynomialFeatures(degree = 2)
    polinomio_3g = PolynomialFeatures(degree = 3)
    #
    resp = 0.0
    #
    if (nome_var_dep == 'Hl'):
        mod = modelos[0]
        resp = mod.predict(polinomio_2g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'Hv'):
        mod = modelos[1]
        resp = mod.predict(polinomio_2g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'y1'):
        mod = modelos[2]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'x1'):
        mod = modelos[3]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    #
    return resp

def f_gera_dados_diag_H_ps(P_eq, npg, lista_componentes,dados):
    '''Função que gera o dataframe com os dados para a preparação do gráfico suporte
        do Método Ponchon-Savarit
        Entrada:
        P_eq = pressão de equilíbrio em mmHg
        npg = quantidade de pontos gerados
        lista_componentes = 
        dados = 
        Saida:
        dados_ps = dataframe com os dados para a construção do gráfico e dos modelos locais 
                para o Método Ponchon-Savarit
    '''
    dados_ps = f_gerar_dados_elv_2c_bd(P_eq, npg, lista_componentes,dados)
    DH_vap = np.zeros((npg,))
    Hig_v  = np.zeros((npg,))
    for i, row in dados_ps.iterrows():
        #print(i, row['T'])
        DH_vap_vet =  f_DHvap_Watson_db(row['T'], lista_componentes, dados)
        y_vez = np.array([row['y1'], (1.0-row['y1'])])
        DH_vap[i] = y_vez @ DH_vap_vet
        Hig_v[i] = f_H_vap_ig_stream(y_vez, 273.15, row['T'], lista_componentes, dados)[0]
    dados_ps['DH_vap_Watson'] = DH_vap
    dados_ps['Hig_v'] = Hig_v
    dados_ps['Hig_l'] = dados_ps['Hig_v'] - dados_ps['DH_vap_Watson']
    return dados_ps


def f_gera_diag_entalpia_ps(H_min_graf, H_max_graf, dados_ps, fig_size):
    ''' Função que gera o gráfico para traçar as linhas do método Ponchon-Savarit:
        Diagrama de entalpia x composição em fração molar
        Entradas:
            H_min_graf = 
            H_max_graf = 
            dados_ps = 
            fig_size = 
        Saida:
            Gráfico com o nome PS_graf
    '''
    fig_ps, ax_ps = plt.subplots(num='PS_graf', figsize=fig_size)
    # pontos e linhas do gráfico
    ax_ps.plot(dados_ps['x1'], dados_ps['Hig_l'], 'b', label='liquido saturado')
    ax_ps.plot(dados_ps['y1'], dados_ps['Hig_v'], 'r', label='vapor saturado')
    #ax_ps.plot(T_BP_comp, DH_vap_comp*4.184, 'go', label='Exp')
    #ax_ps.plot(T_grafico, cp_modelo, 'k', label= 'Mod')
    # Limites dos eixos
    plt.ylim((H_min_graf, H_max_graf))
    # Adicionando texto nos eixos - descrição
    ax_ps.set_xlabel(r'fração molar $x_1$ ou $y_1$')
    ax_ps.set_ylabel(r'Entalpia [$J/mol$]')
    # Adicionando título para a figura
    ax_ps.set_title(r'Diagrama Entalpia x composição')
    # Adicionando um texto
    ax_ps.text(0.4, 13000.0, r'@$P_{eq} = 760.0 \, mmHg$')
    # Adicionando uma legenda
    ax_ps.legend(loc='upper left')
    # Adicionando linha vertical
    #ax_ps.vlines(z[0],H_min_graf, H_max_graf, colors='m', linestyles='dashed')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax_ps.axvline(z[0], colors='k', linestyles=':')
    #ax1.axvline(sol_flash['y_eq'][0])
    # Adicionando linha horizontal
    #ax1.hlines(T_flash,0, 1, colors='y', linestyles='solid')
    # Adicionando grade
    ax_ps.grid()
    # Ajustando os ticks
    from matplotlib.ticker import (MultipleLocator) #AutoMinorLocator
    ax_ps.xaxis.set_major_locator(MultipleLocator(0.10))
    ax_ps.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax_ps.yaxis.set_major_locator(MultipleLocator(10000.0))
    ax_ps.yaxis.set_minor_locator(MultipleLocator(1000.0))
    ax_ps.tick_params(which='minor', length=4, color='r')
    ax_ps.grid(visible=True, which='major', color='k', linestyle='-')
    ax_ps.grid(visible=True, which='minor', color='lightgray', linestyle='--')
    plt.savefig('diagrma_y_x_ELV.png')
    # plt.show()
    return (fig_ps, ax_ps)


def f_Pinch(z, fv, modelos_ps):
    '''Função que determina o ponto de pinch
        Solução do sistema formado pela equação de flash e a curva de equilíbrio, ou ainda,
        ponto no qual a reta de flash intercepta a curva de equilíbrio
        Entradas:
          z = composição da carga
          fv = fração vaporizada da carga - condição da carga
          modelos_ps = modelos locais polinomiais para as curvas do diagrama de
                    fases e de entalpia x composição
        Saídas: dicionário
          x_p: composição da fase líquida no ponto de pinch/ELV
          y_p: composição da fase vapor no ponto de pinch/ELV
    '''
    q = 1 - fv
    if (fv == 0.0):
      x_eq = z[0]
      y_eq = f_uso_mod_loc_ps(x_eq, 'y1', modelos_ps) 
    else:
      a_flash = -(q/(1 - q))
      b_flash = (1/(1 - q))*z[0]
      #
      def f_sol(x, modelos_ps, a_flash, b_flash):
          residuo = a_flash*x + b_flash - f_uso_mod_loc_ps(x, 'y1', modelos_ps)
          return residuo
      # f_sol(0.3,q,z,mod_yeq, a_flash, b_flash)
      # x_eq <- uniroot(f_sol,interval = c(0,1), q,z,mod_yeq, a_flash, b_flash)$root
      x_guest = 0.5
      x_eq = fsolve(f_sol, x_guest, args=(modelos_ps, a_flash, b_flash))[0]
      y_eq = f_uso_mod_loc_ps(x_eq, 'y1', modelos_ps)
    return {'x_p': x_eq, 'y_p': y_eq}


def f_H_carga(z, fv, modelos_ps):
    ''' Função que calcula a entalpia composta da carga
        Obs.: somente para:  0 <= fv <= 1
    '''
    q = 1 - fv
    res_Pinch = f_Pinch(z, fv, modelos_ps)
    #HFL = f_HLx(res_Pinch$x_p, mod_HLx)
    HFL = f_uso_mod_loc_ps(res_Pinch['x_p'], 'Hl', modelos_ps)
    #HFV <- f_HVy(res_Pinch$y_p, mod_HVy)
    HFV = f_uso_mod_loc_ps(res_Pinch['y_p'], 'Hv', modelos_ps)
    HF  = (1 - q)*HFV + q*HFL
    return {'HF': HF, 'HFV': HFV, 'HFL': HFL}


def f_reta_2p(p1,p2):
    ''' Função que calcula os coeficientes da reta que passa pelos pontos p1 e p2
    '''
    # p1 <- c(y1,x1)
    # p2 <- c(y2,x2)
    y1 = p1[0]
    x1 = p1[1]
    y2 = p2[0]
    x2 = p2[1]
    #
    # coeficiente angular
    #
    a = (y1 - y2)/(x1 - x2)
    #
    # coeficiente linear
    #
    b = (y2*x1 - y1*x2)/(x1 - x2)
    #
    return {'b': b, 'a': a}


def f_mostra_linha(p1, p2, cor, ls, fig_name, fig_ps, ax_ps):
    ''' Função que mostra no gráfico ativo uma linha ligando os pontos p1 e p2
        Entrada:
        p1 = coordenadas do ponto 1 (y1, x1) - tupla
        p2 = coordenadas do ponto 2 (y2, x2) - tupla
        cor = cor da linha
        ls  = line style ('-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed',
                          'dashdot', 'dotted')

        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    '''
    x_v = np.array([p1[1], p2[1]])
    y_v = np.array([p1[0], p2[0]])
    plt.figure(num=fig_name)
    ax_ps.plot(x_v, y_v, color=cor, linestyle=ls, marker='o', 
               markerfacecolor=cor, markersize=5.0)
    # plt.show()
    return fig_ps, ax_ps


def f_mostra_ponto(p1, cor, ls, fig_name, fig_ps, ax_ps):
    ''' Função que mostra no gráfico ativo uma linha ligando os pontos p1 e p2
        Entrada:
        p1 = coordenadas do ponto 1 (y1, x1) - tupla
        cor = cor da linha
        ls  = line style ('-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed',
                          'dashdot', 'dotted')
        fig_name:
        fig_ps:
        ax_ps:

        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    '''
    x_v = np.array([p1[1]])
    y_v = np.array([p1[0]])
    plt.figure(num=fig_name)
    ax_ps.plot(x_v, y_v, color=cor, linestyle=ls, marker='o', 
               markerfacecolor=cor, markersize=5.0)
    # plt.show()
    return fig_ps, ax_ps


def f_marca_ponto(p1, cor, mt, ms, fig_name, fig_ps, ax_ps):
  ''' Marca um ponto no gráfico com as coordenadas (y,x) dadas pela tupla'''
  x_v = np.array([p1[1]])
  y_v = np.array([p1[0]])
  plt.figure(num=fig_name)
  ax_ps.plot(x_v, y_v, color=cor, linestyle='', marker=mt, 
               markerfacecolor=cor, markersize=ms)
  #plt.show()
  return fig_ps, ax_ps


def f_mostra_reta(x_min, x_max, reta, npg, cor, mt, ms, ls, fig_name, fig_ps, ax_ps):
  ''' Constroi no gráfico a reta no intervalo entre x_min e x_max com npg pontos'''
  xg = np.linspace(x_min,x_max,npg)
  yg = reta['a']*xg + reta['b']
  plt.figure(num=fig_name)
  ax_ps.plot(xg, yg, color=cor, linestyle=ls, marker=mt, 
               markerfacecolor=cor, markersize=ms)
  return fig_ps, ax_ps


def f_alpha_I_db(Temp, i, j, lista_componentes, dados):
    ''' Equação de definição da volatilidade relativa - alfa_i,j
          o valor de alfa é calculado nesta versão considerando a 
          Lei de Raoult como modelo de volatilidade.
        Entrada:
          Temp: temperatura em K
          i: componente do numerador
          j: componente do denominador
          lista_componentes:
          dados:
        Saida:
          alpha:  K_i / K_j - adimensional
    '''
    num = f_Pvap_Antoine_db(Temp,lista_componentes[i-1], dados)[0]
    den = f_Pvap_Antoine_db(Temp,lista_componentes[j-1], dados)[0]
    alpha = num / den
    return alpha


def f_alpha_medio_db(T1, T2, i, j, lista_componentes, dados):
    ''' Calcula o valor da volatilidade relativa média em uma faixa de 
          temperatura entre os componentes i e j, presentes no banco de dados
          e constantes na lista_componentes
        Entrada:
          T1: 
          T2: 
          i: 
          j:
          lista_componentes:
          dados:
        Saida:
          am: valor de alfa médio calculado - volatilidade relativa
    '''
    am =  integrate.quad(f_alpha_I_db, T1, T2, args=(i, j, lista_componentes, dados))[0]/(T2-T1)
    return am


def f_M_alpha_medio_db(T1, T2, lista_componentes, dados):
    ''' Monta a matriz com os valor das volatilidades relativas entre todos
          os componetes presentes no sistema
        Entrada:
          T1:
          T2:
          lista_componentes:
          dados:
        Saida:
          M_am: matriz NC x NC com os valores de alfa_i,j
                i: índice da linha - numerador
                j: índioce da coluna - denominador
    '''
    nc = len(lista_componentes)
    M_am = np.zeros((nc,nc))
    for i in range(1,nc+1):
        for j in range(1,nc+1):
            M_am[i-1, j-1] = f_alpha_medio_db(T1, T2, i, j, lista_componentes, dados)
    return M_am


def f_NETS_min_FENSKE(i_LK, i_HK, Rec_D_LK, Rec_B_HK, alpha_M):
    ''' Calcula o NETS-min segundo a equação de Fenske - para uso no método FUG
    '''
    num = Rec_D_LK / (1.0 - Rec_D_LK)
    den = (1.0 - Rec_B_HK)/Rec_B_HK
    NETS_min = np.log(num/den)/np.log(alpha_M[i_LK-1,i_HK-1])
    return NETS_min


def f_phi_Underwood(phi, nF, q, z, i_ref,lista_componentes, dados):
    ''' Equação de Underwood para uso no método FUG
    '''
    TBP = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    T1 = TBP.min()
    T2 = TBP.max()
    alpha_M = f_M_alpha_medio_db(T1, T2, lista_componentes, dados)
    nc = z.shape[0]
    soma = 0.0
    ee   = 1.0e-6
    for i in range(1, nc+1):
        parc = (z[i-1]*nF*alpha_M[i-1,i_ref-1])/(alpha_M[i-1,i_ref-1]-phi+ee)
        soma = soma + parc
    fp = nF*(1.0-q) - soma
    return(fp)


def f_RD_min_Underwood(i_LK, i_HK, nF, nD, q, z, Rec_D_est, lista_componentes, dados):
    ''' Calcula a razão de refluxo mínima pela equação de Underwood
        Entrada:
        Saida:
          RD_min: razão de refluxo mínima
          L_min: vazão de líquido na coluna correspondente à RD_min
          V_min: vazão de vapor na coluna correspondente à RD_min
    '''
    #
    nc = len(lista_componentes)
    TBP = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    T1 = TBP.min()
    T2 = TBP.max()
    alpha_M = f_M_alpha_medio_db(T1, T2, lista_componentes, dados)
    #
    phi_0 = np.mean(np.array([alpha_M[i_HK-1, i_LK-1],alpha_M[i_LK-1, i_LK-1]]))
    #
    i_ref = i_LK
    phi = fsolve(f_phi_Underwood, phi_0, args=( nF, q, z, i_ref,lista_componentes, dados))[0]
    #
    soma = 0.0
    for i in range(1,nc+1):
        parc = (z[i-1]*nF*alpha_M[i-1,i_ref-1]*Rec_D_est[i-1])/(alpha_M[i-1,i_ref-1]-phi)
        soma = soma + parc
    V_min = soma
    #
    L_min = V_min - nD
    RD_min = L_min / V_min
    #
    return{'RD_min': RD_min,
           'L_min': L_min, 
           'V_min': V_min}


def f_y_Gilliland(x):
    ''' Função do modelo empírico de Gillinda para uso no método FUG
    '''
    y = 0.0
    if ((x >= 0.0)&(x <= 0.01)):
        y = 1.0 - 18.5715 * x
    if ((x > 0.01)&(x <= 0.90)):
        y = 0.545827 - 0.591422*x + (0.002743 / x)
    if ((x > 0.90)&(x <= 1.0)):
        y = 0.16595 - 0.16595*x
    return y


def f_NETS_and_j_carga_Gilliland(i_LK, i_HK, z, f_RD, RD_min, NETS_min, xD, 
                                 lista_componentes, dados):
    ''' Método FUG - Equação de Gilliland
          Determinação do NETS e do prato de carga
        Entrada:
          i_LK: índice do componete chave leve
          i_HK: índice do componete chave pesado
          z: composição da corrente de alimentação - carga
          f_RD: razão entre a razão de refluxo e a razão de refluxo mínima
          RD_min: razão de refluxo mínima
          NETS_min: 
          xD: composição da corrente de topo D
          lista_componentes:
          dados: 
        Saída: 
          NETS: número de estágio teóricos no sistema necessários para a 
            separação
          j_carga: localização do estágio ou prat no qual a carga deverá ser 
            introduzida na coluna
    '''
    TBP = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    T1 = TBP.min()
    T2 = TBP.max()
    alpha_M = f_M_alpha_medio_db(T1, T2, lista_componentes, dados)
    x_Gill = (f_RD - 1.0) / ((1.0/RD_min) + f_RD)
    y_Gill = f_y_Gilliland(x_Gill)
    NETS = (NETS_min - y_Gill)/(1.0 - y_Gill)
    num = np.log((xD[i_LK-1]/xD[i_HK-1])/(z[i_LK-1]/z[i_HK-1]))
    den = np.log(alpha_M[i_LK-1,i_HK-1])
    j_carga_min = num / den
    j_carga = NETS * ( j_carga_min/ NETS_min)
    j_carga = np.floor(j_carga)
    return {'NETS': NETS,
            'j_carga': j_carga}


def f_PS_linha_op_eq(k,x_ini,x_fin,linha_est,modelos_ps):
    ''' Função que acha o ponto no qual a reta de operação corta a linha de equilíbrio
    '''
    b_linha = linha_est[k,1] 
    a_linha = linha_est[k,0]
    #
    def f_sol_loe(x, a_linha, b_linha, modelos_ps):
      residuo = a_linha*x + b_linha - f_uso_mod_loc_ps(x, 'Hv', modelos_ps)
      return residuo
    #
    x_guest = (x_ini + x_fin)/2.0
    x_sol = fsolve(f_sol_loe, x_guest, args=(a_linha, b_linha, modelos_ps))[0]
    return x_sol


def f_phi_gen(y, T, P):
    ''' Gera os coeficientes de fugacidade de um sistema correspondente a 
          vapor ideal, ou seja: phi_i = 1.0
    '''
    nc = y.shape[0]
    gama = np.zeros((nc,))
    for i in range(0,nc):
      gama[i] = 1.0
    return gama


def f_Lamb_W(T,nc,a_W,Vm):
    ''' Calcula o valor de Lambda da função de Wilson de acordo com a
          equação 12.24 do SVNA
        T   = temperatura em K
        nc  = número de componentes
        a_W = matriz com os parâmetros de interação binária de Wilson (nc x nc)
        Vm  = vetor com os volumes molares dos componentes (nc)
    '''
    R = 1.987 #cal/mol/K
    Lamb_W = np.zeros((nc,nc))
    for i in range(0,nc):
      for j in range(0,nc):
        if (i != j):
          fat1 = Vm[i]/Vm[j]
          fat2 = np.exp(-a_W[i,j]/(R*T))
          Lamb_W[i,j] = fat1 * fat2
        else:
          Lamb_W[i,j] = 1.0
    return(Lamb_W)


def f_gama_Wilson(x, T, a_W, Vm):
    '''Calcula o coeficiente de atividade gama de acordo com a 
         equação 12.23 do SVNA
       x   = vetor das composições da fase líquida em fração molar (nc)
       T   = temperatura em K
       a_W = matriz com os parâmetros de interação binária de Wilson (nc x nc)
       Vm  = vetor com os volumes molares dos componentes (nc)
    '''
    nc = x.shape[0]
    Lamb_W = f_Lamb_W(T,nc,a_W,Vm)
    gama = np.zeros((nc,))
    for i in range(0,nc):
      sum_1 = x @ Lamb_W[i,:]
      sum_2 = 0.0
      for k in range(0,nc):
        sum_3 =  x @ Lamb_W[k,:]
        sum_2 = sum_2 + ((x[k]*Lamb_W[k,i])/sum_3)
      gama[i] = np.exp(1.0 - np.log(sum_1) - sum_2)
    return gama


def f_K_Raoult_mod_Wilson_01_db(T, P, x, y, a_W, Vm, lista_componentes, dados):
    '''Calcula a volarilidade K utilizando os valores da gama do modelo
         do modelo de Wilson e phi da fase gasosa igal a 1.0 - gás ideal
       T   = temperatura em K
       P   = pressão em mmHg
       x   = vetor das composições da fase líquida em fração molar (nc)
       y   = vetor das composições da fase vapor   em fração molar (nc)
       a_W = matriz com os parâmetros de interação binária de Wilson (nc x nc)
       Vm  = vetor com os volumes molares dos componentes (nc)
       lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
       dados - dataframe com os dados do databank
    '''
    nc = len(lista_componentes)
    K = np.zeros((nc,))
    gama = f_gama_Wilson(x, T, a_W, Vm)
    phi  = f_phi_gen(y,T,P)
    for i in range(0,nc):
      num = gama[i] * f_Pvap_Antoine_db(T, lista_componentes[i],dados)[0]
      den = phi[i]  * P
      K[i] = num / den
    return K


def f_res_RR_flash_nid_01_db(fv, z, x, y, P, Temp, a_W, Vm, lista_componentes, dados):
    ''' Função que determina o resíduo da equação de Rachford-Rice para o flash
        multicomponente na solução para encontrar fv (fração vaporizada da carga)
        Obs.: Utilizando a equação de Raoult modificada e com toda a não idealidade
              na fase líquida usando como modelo de Wilson de G_E para o cálculo
              de gama
      Entrada:
      fv - fração vaporizada da carga - variável implícita
      z - composição da carga em fração molar
      P - pressão do flash em mmHg
      Temp - temperatura do flash em K
      a_W = matriz com os parâmetros de interação binária de Wilson (nc x nc)
      Vm  = vetor com os volumes molares dos componentes (nc)
      lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
      dados - dataframe com os dados do databank
      Saidas:
      res - resíduo na busca da solução - res = 0 -> solução
    '''
    nc = len(lista_componentes)
    if (type(fv) == float):
      fv = np.array([fv])
    nr = len(fv)
    K_comp = f_K_Raoult_mod_Wilson_01_db(Temp, P, x, y, a_W, Vm, lista_componentes, dados)
    M_parc = np.empty((nr, nc))
    num = z * K_comp
    for i, fv_vez in enumerate(fv):
        den = 1.0 + fv_vez*(K_comp - 1.0)
        M_parc[i,:] = num / den
    res = 1.0 - np.sum(M_parc, axis=1)
    return res


def f_sol_RR_flash_nid_01_db(z, P, Temp, a_W, Vm, lista_componentes, dados):
    ''' Função que resolve a equação de Rachford-Rice e encontra a fv 
        (fração vaporizada da carga)
        Obs.: Utilizando a equação de Raoult modificada e com toda a não idealidade
              na fase líquida usando como modelo de Wilson de G_E para o cálculo
              de gama
        Entrada:
        z - composição da carga em fração molar
        P - pressão do flash em mmHg
        Temp - temperatura do flash em K
        a_W = matriz com os parâmetros de interação binária de Wilson (nc x nc)
        Vm  = vetor com os volumes molares dos componentes (nc)
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas: {dicionário}
        fv_flash - fração vaporizada - solução do flash
        x_eq - composição do líquido no equilíbrio
        y_eq - composição do vapor no equilíbrio
        K_comp - volatilidade dos componentes
        alpha_comp - volatilidade relativa em relação ao componente chave pesado (i_chk)
    '''
    sol_P_bo = f_calculo_PbPo_db('P', Temp, z, lista_componentes, dados)
    P_pb = sol_P_bo[0]
    P_po = sol_P_bo[1]
    fv_guest = (P_pb - P)/ (P_pb - P_po)
    # Estimativa do valor inicial das composições
    sol_flash_ideal = f_sol_RR_flash_db(z, P, Temp, lista_componentes, dados)
    x0 = sol_flash_ideal['x_eq']
    y0 = sol_flash_ideal['y_eq']
    #
    tol_erro_medio = 1.0e-5
    erro_medio = 1.0e5
    k_it = 1
    #
    while (erro_medio > tol_erro_medio):
      fv_flash = fsolve(f_res_RR_flash_nid_01_db, fv_guest, 
                      args=(z, x0, y0, P, Temp, a_W, Vm, 
                      lista_componentes, dados))[0]
      K_comp = f_K_Raoult_mod_Wilson_01_db(Temp, P, x0, y0, a_W, Vm,
                                            lista_componentes, dados)
      num = z * K_comp
      den = 1.0 + fv_flash*(K_comp - 1.0)
      y_eq = num / den
      x_eq = y_eq / K_comp
      erro_medio_x = np.mean(np.abs(x0 - x_eq))
      erro_medio_y = np.mean(np.abs(y0 - y_eq))
      erro_medio = (erro_medio_x + erro_medio_y)/2.0
      print(k_it, erro_medio)
      k_it += 1
      x0 = x_eq.copy()
      y0 = y_eq.copy()
      fv_guest = fv_flash
    #
    # Finalização
    #
    i_chk = np.argmin(K_comp)
    alpha_comp = K_comp/K_comp[i_chk]
    #
    return {'fv_flash':   fv_flash, 
            'x_eq':       x_eq, 
            'y_eq':       y_eq, 
            'K_comp':     K_comp,
            'alpha_comp': alpha_comp}


def f_Vol_molar_db(lista_componentes, dados):
    ''' Calcula o volume molar dos componentes da lista fornecida utilizando a
          massa específica da fase líquida e a massa molar constante no
          banco de dados.
        Entrada:
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle (nc)
        dados             - dataframe com os dados do databank
        Saida:
        V_m - vetor com os volumes molares da fase líquida em cm3/mol (nc)
    '''
    V_m = dados[dados['num'].isin(lista_componentes)]['molar_mass'].to_numpy()/\
      dados[dados['num'].isin(lista_componentes)]['liq_density'].to_numpy()
    return V_m


def f_Pvap_Antoine_vetor_db(Temp, lista_componentes, dados):
    ''' Calcula a pressão de vapor utilizando o modelo de Antoine para
          todos os componentes da lista_componentes utilizando as constantes
          do banco de dados
        Entrada:
        Temp - temperatura em K
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle (nc)
        dados             - dataframe com os dados do databank
        Saida:
        Pvap_comp - vetor com as P_vap dos componentes em mmHg (nc)
    '''
    nc = len(lista_componentes)
    Pvap_comp = np.zeros((nc,))
    i = 0
    for i_comp in lista_componentes:
      Pvap_comp[i] = f_Pvap_Antoine_db(Temp, i_comp, dados)[0]
      i += 1
    return Pvap_comp


def f_Teq_Antoine_db(Pvap, i_comp, dados):
    #import numpy as np
    ''' Função que calcula a temperatura correspondente a uma dada pressão de vapor, 
          segundo a equação de Antoine, para o componente i_comp presente no 
          databank_properties.pickle.
        Equação de Antoine: Pvap = exp(A - B /(Temp + C)), com: 
          [Temp] = K
          [Pvap] = mmHg
        Entrada (argumentos da função):
          Pvap - pressão de vapor do i_comp em mmHg na qual se deseja Teq
          i_comp = inteiro que corresponde ao número do componente no banco de dados
               campo/column/key = 'num'
          dados  = pandas dataframe com os dados lidos do arquivo databank_properties.pickle
      Saida: tupla
          Teq   = temperatura em K correspondente a Pvap
          par = dicionário com os parâmetros A, B e C da equação de Antoine
    '''
    # param <- as.numeric(param)
    par_array = np.array(dados[dados['num'] == i_comp][['pvap_a','pvap_b','pvap_c']])[0]
    par = {'a': par_array[0], 'b': par_array[1], 'c': par_array[2]}
    a = par['a']
    b = par['b']
    c = par['c']
    #Pvap = np.exp(a - b/(Temp + c))
    Teq = (b/(a - np.log(Pvap))) - c
    # attr(x = Teq, which = "units") <- "K"
    return Teq, par

def f_Dist_Componentes_Fenske(i_LK, i_HK, Rec_D_LK, Rec_B_HK, z, nF, 
                                lista_componentes, dados):
    ''' Função que resolve as equações de Fenske
        Entradas:
          i_LK     = posição do compoente chave leve na lista_componentes
          i_HK     = posição do compoente chave pesado na lista_componentes
          Rec_D_LK = recuperação do componente chave leve no destilado de topo
          Rec_B_HK = recuperação do componente chave pesado no destilado de fundo
          nF       = vazão molar de alimentação
          lista_componentes = lista com os componetes presentes
          dados             = pandas dataframe com os dados lidos do arquivo 
                                databank_properties.pickle
        Saídas:
          NETS_min = NETS mínimo
          Rec_D_est = vetor com recuperação dos componentes no destilado de topo D
          xD        = composição em fração molar da corrente de topo D
          nD        = vazão molar da corrente de topo D
          nB        = vazão molar da corrente de fundo B
    '''
    nc = len(lista_componentes)
    TBP = dados[dados['num'].isin(lista_componentes)]['boiling_point'].to_numpy()
    T1 = TBP.min()
    T2 = TBP.max()
    alpha_M = f_M_alpha_medio_db(T1, T2, lista_componentes, dados)
    NETS_min = f_NETS_min_FENSKE(i_LK, i_HK, Rec_D_LK, Rec_B_HK, alpha_M)
    #
    # Equação de Fenske para distribuição dos componentes não-chave
    #
    Rec_D_est = np.zeros((nc,))
    for i in range(1,nc+1):
        if (i == i_LK):
            Rec_D_est[i-1] = Rec_D_LK
        elif (i == i_HK):
            Rec_D_est[i-1] = 1 - Rec_B_HK
        elif ((i != i_LK)&(i != i_HK)):
            num = (alpha_M[i-1,i_HK-1])**NETS_min
            parc1 = Rec_B_HK/(1- Rec_B_HK)
            den = parc1 + num
            Rec_D_est[i-1] = num / den
    #
    # Calculo da composição no destilado de topo e da vazão de destilado no topo
    #
    def fun_senl(x,b):
        neq = len(x)
        y = np.zeros((neq,))
        #
        nc = neq - 1
        xD = x[1-1:nc]
        nD = x[neq-1]
        #
        for i in range(1,nc+1):
            y[i-1] = xD[i-1]*nD - b[i-1]
            y[neq-1] = xD.sum() - b[neq-1]
        #
        return y
    #
    # vetor independente do sistema
    #
    b = np.zeros((nc+1,))
    for i in range(1,nc+1):
        b[i-1] = z[i-1]*nF*Rec_D_est[i-1]
        b[nc+1-1] = 1.0
    #
    # chute inicial para a solução do sistema
    # 
    x0 = np.zeros((nc+1,))
    x0[1-1:nc] = 1.0/nc
    x0[nc+1-1] = nF/2.0
    # Solução do sistema de equações para obter
    #   - composição do destilado de topo
    #   - vazão molar do destilado de topo
    #from scipy import optimize
    #sol_senl = fsolve(fun_senl, x0, epsfcn = 1.0e-4,
    #             args=(b,), xtol=1.0e-8)
    sol_senl = root(fun_senl, x0, method='lm',
                 args=(b,), tol=1.0e-8)
    #xD = sol_senl[0:nc]
    #nD = sol_senl[-1]
    xD = sol_senl.x[0:nc]
    nD = sol_senl.x[-1]
    nB = nF - nD
    return {'NETS_min':NETS_min, 
            'Rec_D_est':Rec_D_est,
            'xD': xD,
            'nD':nD,
            'nB':nB}


def f_sol_RR_flash_jvat_db(z, P, Temp, lista_componentes, dados):
    ''' Função que resolve a equação de Rachford-Rice e encontra a fv 
        (fração vaporizada da carga)
        Entrada:
        z - composição da carga em fração molar
        P - pressão do flash em mmHg
        Temp - temperatura do flash em K
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
        Saidas: {dicionário}
        fv_flash - fração vaporizada - solução do flash
        x_eq - composição do líquido no equilíbrio
        y_eq - composição do vapor no equilíbrio
        K_comp - volatilidade dos componentes
        alpha_comp - volatilidade relativa em relação ao componente chave pesado (i_chk)
    '''
    try:
        if not lista_componentes:
            raise Exception('Rodar lista de componentes')
        if dados.empty != False:
            raise Exception('Rodar DataFrame de dados')
        #
        T_Pb, T_Po, P_vap_eb_comp = f_calculo_PbPo_db('T', P, z, lista_componentes, dados)
        #
        if (T_Pb - T_Po) <= 5:
            print(f'Cuidado, seu ponto de bolha e orvalho estão próximos: T_Pb:{T_Pb:.2f}  T_Po:{T_Po:.2f}')
        if Temp < T_Pb:
            fv_flash = 0.0
            print('Abaixo do ponto de bolha')
        elif Temp > T_Po:
            fv_flash = 1.0
            print('Acima do ponto de orvalho')
        elif Temp < 0.4 * (T_Po - T_Pb) + T_Pb:
            fv_guest = 0.3
            fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
        elif (Temp < 0.7 * (T_Po - T_Pb) + T_Pb) & (Temp > 0.4 * (T_Po - T_Pb) + T_Pb):
            fv_guest = 0.5
            fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
        elif Temp >= 0.7 * (T_Po - T_Pb) + T_Pb:
            fv_guest = 0.7
            fv_flash = fsolve(f_res_RR_flash_db, fv_guest, args=(z, P, Temp, lista_componentes, dados))[0]
        K_comp = f_K_Raoult_db(Temp, P, lista_componentes, dados)[0]
        num = z * K_comp
        den = 1.0 + fv_flash*(K_comp - 1.0)
        y_eq = num / den
        x_eq = y_eq / K_comp
        i_chk = np.argmin(K_comp)
        alpha_comp = K_comp/K_comp[i_chk]
        return {'fv_flash': fv_flash, 'x_eq': x_eq, 'y_eq': y_eq, 'K_comp': K_comp,
                'alpha_comp': alpha_comp}
    except Exception as e:
        print(e)


def f_gera_diag_T_x_y(P_eq, n_pontos, lista_componentes, dados):
    ''' Função que gera o diagrama T-x-y de uma mistura binária; são traçadas
          as curvas de ponto de bolha e ponto de orvalho.
        P_eq - pressão de equilíbrio em mmHg
        n_pontos - número de pontos para a construção das curvas
        lista_componentes - lista com os números de identificação dos componentes
                          do sistema correspondente ao 
                          databank_properties.pickle
        dados - dataframe com os dados do databank
    '''
    dados_elv = f_gerar_dados_elv_2c_bd(P_eq, n_pontos, lista_componentes, dados)
    # Fazendo o gráfico
    fig1, ax1 = plt.subplots( figsize =(8,8))
    ax1.plot(dados_elv['x1'], dados_elv['T'], 'b', label='bolha')
    ax1.plot(dados_elv['y1'], dados_elv['T'], 'g', label='orvalho')
    # Adicionando texto nos eixos - descrição
    ax1.set_xlabel('x,y')
    ax1.set_ylabel('T [K]')
    # Adicionando título para a figura
    ax1.set_title('Diagrama T-x-y')
    # Adicionando um texto
    T_pos = 0.995 * dados_elv['T'].max()
    ax1.text(0.5, T_pos, r'@$P_{eq} em mmHg$')
    # Adicionando uma legenda
    ax1.legend()
    # Adicionando linha vertical
    #ax1.vlines(z[0],T_Pb, T_Po, colors='m', linestyles='dashed')
    #ax1.axvline(sol_flash['x_eq'][0])
    #ax1.axvline(sol_flash['y_eq'][0])
    # Adicionando linha horizontal
    #ax1.hlines(T_flash,0, 1, colors='y', linestyles='solid')
    # Adicionando grade
    ax1.grid()
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    ax1.xaxis.set_major_locator(MultipleLocator(0.10))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax1.yaxis.set_major_locator(MultipleLocator(1.00))
    ax1.tick_params(which='minor', length=4, color='r')
    #ax1.grid(True, which='minor')
    return fig1, ax1


def f_visc_liq_db(Temp, param):
  ''' Calcula a viscosidade a partir dos parâmetros b e c para o modelo
         de dependência com a temperatura compatível com o 
         databank_properties.pickle
      Entrada:
        Temp:  temperatura em K
        param: vetor ou lista com os parãmetros b e c do modelo de
                viscosidade em função da temperatura
      Saída:
        visc = Viscosidade do líquido em cP (centi Poise)
  '''
  b = param[0]
  c = param[1]
  visc = 10**(b*((1/Temp)-(1/c)))
  # attr(x = visc, which = "units") <- "cP"
  return visc


def f_visc_liq_comp_db(Temp, num, dados):
  ''' Calcula a viscosidade em cP para o componente num presente no banco de 
        dados databank_properties.pickle.
      Entrada:
        Temp:  temperatura em K
        num:   código numérico do componente no databank_properties.pickle
        dados: dataframe com o banco de dados
      Saída:
        visc = Viscosidade do líquido em cP (centi Poise)
  '''
  # Obtençao dos parâmetros do modelo de viscisidade
  param = dados[dados['num'] == num][['visc_b', 'visc_c']].to_numpy()[0]
  #
  b = param[0]
  c = param[1]
  #
  visc = 10**(b*((1/Temp)-(1/c)))
  # attr(x = visc, which = "units") <- "cP"
  return visc


def f_eq_Harlacher(Pvap, Temp, param):
  ''' Equação implícita do modelo de pressão de vapor de Harlacher

      Não tem aplicação isolada!
  '''
  #
  a = param[0]
  b = param[1]
  c = param[2]
  d = param[3]
  #
  if (Pvap < 1.0e-5):
    parcela = -15.0
  else:
    parcela = np.log(Pvap)
  #
  fun = parcela - a - b/Temp - c*np.log(Temp) - ((d*Pvap)/Temp**2)
  #
  return fun


def f_Pvap_Harlacher_db(Temp, param):
  ''' Calcula a pressão de vapor segundo o modelo de Harlacher
      Emtrada:
        Temp: temperatura em K
        param: vetor ou lista com os parâmetros a, b, c e d do modelo de
                Harlacher em função da temperatura
      Saída:
        Pvap: pressão de vapor do componente em mmHg
  '''
  Pvap_guest = 350.0 # mmHg
  Pvap = root(f_eq_Harlacher, Pvap_guest, method='lm', args=(Temp, param),
              tol=1.0e-8).x[0];
  # attr(x = Pvap, which = "units") <- "mmHg"
  return Pvap


def f_Pvap_Harlacher_comp_db(Temp, num, dados):
  ''' Calcula a pressão de vapor segundo o modelo de Harlacher
      Emtrada:
        Temp: temperatura em K
        num:   código numérico do componente no databank_properties.pickle
        dados: dataframe com o banco de dados
      Saída:
        Pvap: pressão de vapor do componente em mmHg
  '''
  # Obtençao dos parâmetros do modelo de Harlacher
  par_H = dados[dados['num'] == num][['pvap_h_a','pvap_h_b',
                                      'pvap_h_c','pvap_h_d']].to_numpy()[0]
  #
  Pvap_guest = 350.0 # mmHg
  Pvap = root(f_eq_Harlacher, Pvap_guest, method='lm', args=(Temp, par_H),
              tol=1.0e-8).x[0];
  # attr(x = Pvap, which = "units") <- "mmHg"
  return Pvap


def f_nets_min_McCabe(p1, p3, P, n_steps_max, fig_name, fig, ax, lista_comp, dados):
    ''' Cálculo do NETS mínimo segundo o método McCabe-Thiele
    '''
    # Reta bissetriz
    reta_biss = f_reta_2p((0.0,0.0), (1.0,1.0))
    yx_steps = np.zeros((n_steps_max, 2))
    # Gerando um conjunto de dados de equilíbrio que servirá de base para
    #   estimação dos parâmetros dos modelos locais
    npg = 20
    dados_ps = f_gera_dados_diag_H_ps(P, npg, lista_comp,dados)
    # Gerando os modelos locais para os dados de ELV
    modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
    # os degraus para determinação do NETS partem do p1
    yx_steps[0,:] = p1
    # primeiro estágio
    k  = 1 # estágio
    kk = 1 # ponto dos steps
    y_eq_k = yx_steps[kk-1,0]
    x_eq_k = f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
    yx_steps[kk,:] = (y_eq_k, x_eq_k)
    kk = kk + 1
    y_rop_k = reta_biss['a']*x_eq_k +  reta_biss['b']
    yx_steps[kk,:] = (y_rop_k, x_eq_k)
    kk = kk + 1
    k = k + 1
    # demais estágios...
    while (x_eq_k >= p3[1]):
        y_eq_k = yx_steps[kk-1,0]
        x_eq_k = f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
        yx_steps[kk,:] = (y_eq_k, x_eq_k)
        kk = kk + 1
        #
        y_rop_k = reta_biss['a']*x_eq_k +  reta_biss['b']
        #
        yx_steps[kk,:] = (y_rop_k, x_eq_k)
        kk = kk + 1
        k = k + 1
    #
    NETS_min = k - 1
    #
    #plt.figure(num=fig_name);
    ax.step(yx_steps[0:(kk-1),1], yx_steps[0:(kk-1),0], color='red');
    #plt.show()
    #
    return {'NETS_min': NETS_min,
            'fig': fig,
            'ax': ax, 
            'modelos_ps': modelos_ps, 
            'r2_modelos_ps': r2_modelos_ps}


def f_nets_McCabe(p1, p2, p3, p4, p5, P, n_steps_max, fig_name, fig, ax, lista_comp, dados):
    ''' Cálculo do NETS segundo o método McCabe-Thiele, fornece também a localização
        do estágio de alimentação (carga)
    '''
    # Reta ROZER
    reta_ROZER = f_reta_2p(p1,p2)
    # Reta ROZEG
    reta_ROZEG = f_reta_2p(p3,p5)
    #
    # Gerando um conjunto de dados de equilíbrio que servirá de base para
    #   estimação dos parâmetros dos modelos locais
    npg = 20
    dados_ps = f_gera_dados_diag_H_ps(P, npg, lista_comp,dados)
    # Gerando os modelos locais para os dados de ELV
    modelos_ps, r2_modelos_ps = f_gera_mod_locais_ps(dados_ps)
    # Matriz para guardar os pontos dos degraus
    yx_steps = np.zeros((n_steps_max, 2))
    # os degraus para determinação do NETS partem do p1
    yx_steps[0,:] = p1
    # primeiro estágio
    k  = 1 # estágio
    kk = 1 # ponto dos steps
    y_eq_k = yx_steps[kk-1,0]
    x_eq_k = f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
    yx_steps[kk,:] = (y_eq_k, x_eq_k)
    kk = kk + 1
    y_rop_k = reta_ROZER['a']*x_eq_k +  reta_ROZER['b']
    yx_steps[kk,:] = (y_rop_k, x_eq_k)
    kk = kk + 1
    k = k + 1
    # demais estágios...
    j_carga = 0
    while (x_eq_k >= p3[1]):
        y_eq_k = yx_steps[kk-1,0]
        x_eq_k = f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
        yx_steps[kk,:] = (y_eq_k, x_eq_k)
        kk = kk + 1
        #
        if (x_eq_k >= p5[1]):
            y_rop_k = reta_ROZER['a']*x_eq_k +  reta_ROZER['b']
        else:
            y_rop_k = reta_ROZEG['a']*x_eq_k +  reta_ROZEG['b']
            if (j_carga == 0):
                j_carga = k
        #
        yx_steps[kk,:] = (y_rop_k, x_eq_k)
        kk = kk + 1
        k = k + 1
    #
    NETS = k - 1
    #
    #plt.figure(num=fig_name);
    ax.step(yx_steps[0:(kk-1),1], yx_steps[0:(kk-1),0], color='red');
    #plt.show()
    #
    return {'NETS': NETS,
            'j_carga': j_carga,
            'fig': fig,
            'ax': ax, 
            'modelos_ps': modelos_ps, 
            'r2_modelos_ps': r2_modelos_ps,
            'yx_steps': yx_steps}


def f_eq_ELV_alpha(x, alpha):
    ''' Função que calcula y em função de alpha e de x
        Entradas:
        x:     concentração na fase líquida em equilíbrio em fração molar
        alpha: volatilidade relativa em sistema binário
        Saida:
        y:     concentração na fase vapor em equilíbrio em fração molar
    '''
    num = alpha * x
    den = 1.0 + (alpha - 1.0)*x
    y = num / den
    return y


def f_integ_Rayleigh_alpha(x, alpha):
    ''' Integrando da equação de Rayleig sendo y_eq definido por alpha
    '''
    y_eq = f_eq_ELV_alpha(x, alpha)
    fun = 1.0 / ( y_eq -x)
    return fun


def f_FR_Rayleigh_alpha(xR, xR0, alpha):
    ''' Calcula a fração residual e a fração destilada pela equação de Rayleigh
    '''
    fr = np.exp(integrate.quad(f_integ_Rayleigh_alpha, xR0, xR, args=(alpha,))[0])
    fd = 1.0 - fr
    return {'FR': fr, 'FD': fd}


def f_res_FR_Rayleigh_alpha(xR, xR0, FR, alpha):
    ''' Calcula o resíduo na solução da xR_A que corresponde a um valor de FR
          na destilação em batelada usando a equação de Rayleigh definida pelo
          equilibrio representado por alpha
    '''
    FR_calc = f_FR_Rayleigh_alpha(xR, xR0, alpha)
    res = FR - FR_calc['FR']
    return res


def f_calc_xR_dado_FR_Rayleigh_alpha(FR, xR0, alpha):
    ''' Calcula o valor de xR_A dado o valor de FR a partir da soluçãao da
          equação de Rayleigh utilizando a volatilidade relativa alpha
          para representaro ELV em um sistema binário
    '''
    x_guest = 0.50
    xR_A = fsolve(f_res_FR_Rayleigh_alpha, x_guest, args=(xR0, FR, alpha))[0]
    if xR_A < 0.0:
      xR_A = 0.0
    return {'xR_A': xR_A, 'FR': FR, 'FD': (1.0 - FR)}


def f_isot_linear(C_A, K_L):
  ''' Isoterma Linear
      Entradas:
        C_A: concentração de A no equilíbrio
        
        K_L: constante da isoterma linear
      Saidas:
        q_A: concentração de A no adsrovente no equilíbrio em razão mássica
  '''
  q_A = K_L * C_A
  return q_A


def f_isot_Freundlich(C_A, K_F, n):
  ''' Isoterma de Freundlich
      Entradas:
        C_A: concentração de A no equilíbrio
        
        K_L: constante da isoterma de Freundlich
        
        n:   expoente da isoterma de Freundlich
      Saidas:
        q_A: concentração de A no adsrovente no equilíbrio em razão mássica
  '''
  q_A = K_F * np.power(C_A,n)
  return q_A


def f_isot_Langmuir(C_A, K_ads, q_sat):
  ''' Isoterma de Langmuir
      Entradas:
        C_A: concentração de A no equilíbrio
        
        K_ads: constante da equilíbrio da adsorção
        
        q_sat: concentração de saturação no adsorvente
      Saidas:
        q_A: concentração de A no adsrovente no equilíbrio em razão mássica
  '''
  q_A = q_sat * (K_ads*C_A) / (1.0 + (K_ads*C_A))
  return q_A


def f_res_bal_mat_adsorcao_batelada(x, i_isoterma, dados_ads_bat, param_isot):
    '''
      Entradas:
        x: Variáveis do problema:
          x[0] --> q_A
          
          x[1] --> C_A_eq
        i_isoterma: número inteiro de 1 a 3 indicando o tipo de isoterma usada
          1 --> Linear
          
          2 --> Freundlich
          
          3 --> Langmuir
        dados_ads_bat: dicionário com os dados do problema de adsorção em batelada, sendo as
                       seguintes keys:
          C_A_0:
          
          m_ADS:
          
          q_0_A:
          
          V_sol:
        param_isot:  dicionário com os parâmetros das isotermas, com as seguintes keys:
          K_L:
          
          K_F:
          
          n:
          
          K_ads:
          
          q_sat:
      Saidas:
        res: resíduo da equação do balanço de massa na forma de vetor
    '''
    n_var = x.shape[0]
    res = np.zeros((n_var,))
    #
    q_A    = x[0]
    C_A_eq = x[1]
    #
    C_A_0 = dados_ads_bat['C_A_0']
    m_ADS = dados_ads_bat['m_ADS']
    q_0_A = dados_ads_bat['q_0_A']
    V_sol = dados_ads_bat['V_sol']
    #
    res[0] = C_A_0*V_sol + q_0_A*m_ADS - C_A_eq*V_sol - q_A*m_ADS
    #
    # Dependendo da isoterma
    #
    if (i_isoterma == 1):
      K_L = param_isot['K_L']
      res[1] = q_A - f_isot_linear(C_A_eq,K_L)
    if(i_isoterma == 2):
      K_F = param_isot['K_F']
      n   = param_isot['n']
      res[1] = q_A - f_isot_Freundlich(C_A_eq, K_F, n)
    if(i_isoterma == 3):
      K_ads = param_isot['K_ads']
      q_sat = param_isot['q_sat']
      res[1] = q_A - f_isot_Langmuir(C_A_eq,K_ads,q_sat)
    return res


def f_Y_ELV_absorcao(X,H):
  ''' Calcula a concentração em razão molar em equilíbrio na fase gasosa para o ELV descrito pela 
        equação Henry na forma y_A = H * x_A
      Entrada:
        X: concentracão no equilíbrio na fase líquida em razão molar
        H: constante de Henry na forma de y = H*x
      Saida:
        Y: concentração no equilíbrio na fase gasosa em razão molar
  '''
  num = H*X
  den = X + 1.0 - H*X
  Y = num/den
  return Y


def f_X_ELV_absorcao(Y,H):
  ''' Calcula a concentração em razão molar em equilíbrio na fase líquida para o ELV descrito pela 
        equação Henry na forma y_A = H * x_A
      Entrada:
        Y: concentração no equilíbrio na fase gasosa em razão molar
        H: constante de Henry na forma de y = H*x
      Saida:
        X: concentracão no equilíbrio na fase líquida em razão molar
  '''
  num = Y
  den = H - Y + H*Y
  X = num/den
  return X


def f_vazao_solv_min_absorcao(par, H):
    '''Função para o cálculo da vazão mínima de solvente na absorção
        Entrada:
            par: dicionário com no mínimo os seguintes parâmetros:
                'G_I':
                'Y_A_T':
                'Y_A_F':
                'X_A_T':
            H: constante de Henry na forma y = H*x
        Saida:
            L_S_min: Vazão de solvente mínima correpondete ao pinch de saturação do 
                solvente na saída de fundo da coluna
    '''
    #
    G_I   = par['G_I']
    Y_A_T = par['Y_A_T']
    Y_A_F = par['Y_A_F']
    X_A_T = par['X_A_T']
    #
    # Calculo do X_sat correspondente ao pinch
    #
    X_A_sat = f_X_ELV_absorcao(Y_A_F,H)
    #
    # Calculo do L_S_min
    #
    num = Y_A_F   - Y_A_T
    den = X_A_sat - X_A_T
    L_S_min = G_I * (num / den)
    #
    return L_S_min


def f_Y_reta_operacao_absorcao(X,par):
    ''' Função que calcula os valores de Y na Reta de operação'''
    #
    L_S   = par['L_S']
    G_I   = par['G_I']
    Y_A_T = par['Y_A_T']
    X_A_T = par['X_A_T']
    #
    Y_RO = (L_S / G_I)* X + (Y_A_T*G_I - X_A_T*L_S)/(G_I)
    return Y_RO


def f_X_reta_operacao_absorcao(Y,par):
    '''Função que calcula os valores de X na Reta de operação'''
    #
    L_S = par['L_S']
    G_I = par['G_I']
    Y_A_T = par['Y_A_T']
    X_A_T = par['X_A_T']
    #
    X_RO = (G_I / L_S)*(Y - (Y_A_T*G_I - X_A_T*L_S)/(G_I))
    return X_RO


def f_nets_McCabe_absorcao(p_topo, p_fundo, n_steps_max, fig_name, fig, ax, H, par_abs):
    ''' Cálculo do NETS mínimo segundo o método McCabe-Thiele para uma coluna de absorcao com o equilíbrio
          ELV representado pela equação de Henry na forma de y = H*x  (fração molar).
          A base do cálculo é o ELV no espaço de variáveis de razão molar nas fases liquida e vapor
    '''
    #
    yx_steps = np.zeros((n_steps_max, 2))
    yx_steps[0,:] = p_fundo
    # primeiro estágio
    k  = 1 # estágio
    kk = 1 # ponto dos steps
    #y_eq_k = yx_steps[kk-1,0]
    #x_eq_k = f_X_ELV_absorcao(y_eq_k,H) #f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
    x_eq_k = yx_steps[kk-1,1]
    y_eq_k = f_Y_ELV_absorcao(x_eq_k,H)
    print(k,(y_eq_k, x_eq_k))
    yx_steps[kk,:] = (y_eq_k, x_eq_k)
    kk = kk + 1
    #y_rop_k =  f_Y_reta_operacao_absorcao(x_eq_k,par_abs) # reta_biss['a']*x_eq_k +  reta_biss['b']
    #yx_steps[kk,:] = (y_rop_k, x_eq_k)
    x_rop_k =  f_X_reta_operacao_absorcao(y_eq_k,par_abs)
    yx_steps[kk,:] = (y_eq_k, x_rop_k)
    #
    kk = kk + 1
    k = k + 1
    # demais estágios...
    while ((y_eq_k >= p_topo[0])&(x_eq_k >= p_topo[1])):
        #y_eq_k = yx_steps[kk-1,0]
        #x_eq_k = f_X_ELV_absorcao(y_eq_k,H) #f_uso_mod_loc_ps(y_eq_k, 'x1', modelos_ps)
        x_eq_k = yx_steps[kk-1,1]
        y_eq_k = f_Y_ELV_absorcao(x_eq_k,H)
        print(k,(y_eq_k, x_eq_k))
        yx_steps[kk,:] = (y_eq_k, x_eq_k)
        kk = kk + 1
        #
        #y_rop_k = f_Y_reta_operacao_absorcao(x_eq_k,par_abs) # reta_biss['a']*x_eq_k +  reta_biss['b']
        #yx_steps[kk,:] = (y_rop_k, x_eq_k)
        x_rop_k =  f_X_reta_operacao_absorcao(y_eq_k,par_abs)
        yx_steps[kk,:] = (y_eq_k, x_rop_k)
        #
        kk = kk + 1
        k = k + 1
    #
    NETS = k - 1
    #
    plt.figure(num=fig_name);
    ax.step(yx_steps[0:(kk-1),1], yx_steps[0:(kk-1),0], color='red');
    #plt.show()
    #
    return {'NETS': NETS,
            'fig': fig,
            'ax': ax,
            'yx_steps': yx_steps}


def f_Y_inu_cgpc(X_cgpc):
    '''  Calcula a ordenada Y_cgpc na condição de inundação para um dado valor de X_cgpc
        Entrada:
            X_cgpc:
        Saida:
            Y_inu_cgpc:
    '''
    #
    pp_CGPC = np.array([-0.6847027, -1.0860630, -0.2980029])
    #
    c = pp_CGPC[0]
    b = pp_CGPC[1]
    a = pp_CGPC[2]
    #
    logX   = np.log10(X_cgpc)
    logY   = a * (logX)**2 + b*logX + c
    Y_inu_cgpc = 10.0**(logY)
    return Y_inu_cgpc
  

def f_Y_cgpc_coord(phi_g, rho_l, rho_g, visc_l_cp, cf):
    ''' Calcula o fluxo de gás na condição de inundação correspondente a abcissa X_cgpc
        Entrada:
            phi_g: fluxo mássico de gás em kg/(m2.s)
            rho_l: massa específica do líquido em kg/m3
            rho_g: massa específica do gás em kg/m3
            visc_l_cp: viscosidade do líquido em cPoise
            cf: fator de empacotamento do recheio
        Saida:
            Y_cgpc: ordenada do gráfico de CGPC
    '''
    # Fator de correção
    c = 2.994
    # Massa específica da água a 25ºC
    rho_H2O = 997.0 # kg/m^3
    psi = rho_H2O/rho_l
    #
    Y_cgpc = (phi_g**2*cf*psi*(visc_l_cp**0.2))/(rho_g*rho_l*c)
    #
    #phi_inu = ((Y_cgpc*rho_g*rho_l*c)/(cf*psi*(visc_l_cp**0.2)))**(1/2)
    #phi_inu # kg/(m^2*s)
    #
    return Y_cgpc

  
def f_phi_inundacao_cgpc(X_cgpc, rho_l, rho_g, visc_l_cp, cf):
    ''' Calcula o fluxo de gás na condição de inundação correspondente a abcissa X_cgpc
        Entrada:
            X_cgpc:
            rho_l:
            rho_g:
            visc_l_cp:
        Saida:
            phi_inu
    '''
    # Fator de correção
    c = 2.994
    # Massa específica da água a 25ºC
    rho_H2O = 997.0 # kg/m^3
    psi = rho_H2O/rho_l
    #
    Y_cgpc = f_Y_inu_cgpc(X_cgpc)
    #
    phi_inu = ((Y_cgpc*rho_g*rho_l*c)/(cf*psi*(visc_l_cp**0.2)))**(1/2)
    phi_inu # kg/(m^2*s)
    #
    return phi_inu


def func_1_m_y_ml(y_1, y_2):
    ''' Função para o cálculo de (1-y)ml
    '''
    num = (1.0 - y_1) - (1.0 - y_2)
    den = np.log((1.0 - y_1)/(1.0 - y_2))
    resp1_y_ml = num / den
    return(resp1_y_ml)


def func_ymysat_ml(y_1,y_1_sat,y_2,y_2_sat):
    ''' Função para o cálculo de (y-y*)ml
    '''
    num = (y_1 - y_1_sat) - (y_2 - y_2_sat)
    den = np.log((y_1 - y_1_sat)/(y_2- y_2_sat))
    respy_y_ml = num / den
    return(respy_y_ml)


def f_res_Z(Z, beta, q, epsil, sigma, koeq):
  res = 1.0e+32
  if (Z == 0.0):
    Z = 1.0e-4
  if (koeq == 1):
    # Eq 3.52
    res = 1.0 + beta - q*beta*(Z-beta)/((Z+epsil*beta)*(Z+sigma*beta)) - Z
  elif (koeq == 2):
    # Eq. 3.56
    res = beta + (Z+epsil*beta)*(Z+sigma*beta) *((1.0 + beta - Z)/(q*beta)) - Z
  return res


def f_sol_Z(Temp,P,y,EoS,koeq,lista_componentes,dados):
  '''Função que calcula o fator de compressibilidade a partir de uma equação de estado 
     a escolha.
     Entradas:
       Temp: Temperatura (K)
          P: Pressão (bar)
          y: Composição em fração molar - componentes na mesma ordem que em lista_componentes
        EoS: valor de 1 a 4 que define a equação de estado a ser usada, em que:
             1 - van der Waals (vdW)
             2 - Redlich/Kwong (RK)
             3 - Soave/Redlich/Kwong (SRK)
             4 - Peng/Robinson (PR)
       koeq: coeficiente que indica qual equação do SVNA sera utilizada, em que:
             1 - equação 3.52 SVNA
             2 - equação 3.56 SVNA
             Deve ser utilizada a equação que convergir melhor
      lista_componentes:
      dados:
  '''
  # Parâmetros
  R = 8.3144621 # J/(K.mol)
  f_bar_Pa = 1.0e5/1.0
  P_Pa = P * f_bar_Pa
  # Massa molar dos componentes em kg/kmol
  MM_comp = dados[dados['num'].isin(lista_componentes)]['molar_mass'].to_numpy()
  # Propriedades críticas dos componentes
  omega = dados[dados['num'].isin(lista_componentes)]['acentric_factor'].to_numpy()
  Tc    = dados[dados['num'].isin(lista_componentes)]['critical_temp'].to_numpy()
  Pc    = dados[dados['num'].isin(lista_componentes)]['critical_pressure'].to_numpy()
  # Propriedades críticas da mistura
  omega_m = y @ omega
  Tc_m = y @ Tc
  Pc_m = y @ Pc
  # Massa molar média da mistura
  MM_m = y @ MM_comp
  # Propriedades reduzidas
  Tr = Temp/Tc_m
  Pr = P/Pc_m
  # Parâmetros das EoS cúbicas - Tabela 3.1 p. 72 SVNA
  if EoS == 1:
    name_EoS = 'VdW'
    alpha = 1.0
    sigma = 0
    epsil = 0
    Omega = 1/8
    Psi   = 27/64
  if EoS == 2:
    name_EoS = 'RK'
    alpha = Tr**(-1/2)
    sigma = 1.0
    epsil = 0
    Omega = 0.08664
    Psi   = 0.42748
  if EoS == 3:
    name_EoS = 'SRK'
    alpha = (1.0 + (0.480 + 1.574*omega_m - 0.176*omega_m**2)*(1.0 - Tr**(1/2)))**2
    sigma = 1.0
    epsil = 0
    Omega = 0.08664
    Psi   = 0.42748
  if EoS == 4:
    name_EoS = 'PR'
    alpha = (1.0 + (0.37464 + 1.54226*omega_m - 0.26992*omega_m**2)*(1.0 - Tr**(1/2)))**2
    sigma = 1.0
    epsil = 0
    Omega = 0.07780
    Psi   = 0.45724
  #  
  beta = Omega*Pr/Tr          # equação 3.53 SVNA
  q    = Psi*alpha/(Omega*Tr) # equação 3.54 SVNA
  # Cálculo do fator de compressibilidade Z
  sol_Z = fsolve(f_res_Z, 0.5, full_output=True,
             args=(beta, q, epsil, sigma, koeq), xtol=1.0e-8)
  Z     = sol_Z[0][0]
  # Cálculo do volume molar
  V_m3_kmol = (Z*R*Temp/P_Pa) * 1000.0
  V_m3_kg   = V_m3_kmol / MM_m
  rho = 1.0 / V_m3_kg
  #
  saida = {'Z': Z, 'Nome_EoS': name_EoS, 'Tr': Tr, 'Pr': Pr, 'T_K': Temp,
           'P_bar': P, 'alpha': alpha, 'sigma': sigma, 'epsilon': epsil,
           'Omega': Omega, 'Psi': Psi, 'beta': beta, 'q': q,
           'V_m3_kmol': V_m3_kmol, 'rho_kg_m3': rho,
           'Tc_K': Tc, 'Pc_bar': Pc, 'omega': omega, 'MM_kg/kmol': MM_comp,
           'Tc_m': Tc_m, 'Pc_m': Pc_m, 'omega_m': omega_m, 'MM_m': MM_m}
  return saida


def f_h_liq_stream(z, T_ref, Temp, lista_componentes, dados):
  h_vap_stream = f_H_vap_ig_stream(z, T_ref, Temp, lista_componentes, dados)[0]
  DHvap_stream = z @ f_DHvap_Watson_db(Temp, lista_componentes, dados)
  h_liq_stream = h_vap_stream - DHvap_stream
  return h_liq_stream


def f_busca_T_flash_dado_fV(f_V, P, z, lista_componentes, dados):
    res = [0.0, 0.0, 0.0]
    res = f_calculo_PbPo_db('T', P, z, lista_componentes, dados)
    T_pb = res[0]
    T_po = res[1]
    T_guest = T_pb + f_V*(T_po - T_pb)
    def f_res_fV(T, f_V, z, P, lista_componentes, dados):
        #resul = f_sol_RR_flash_jvat_db(z, P, T, lista_componentes, dados);
        resul = f_sol_RR_flash_db(z, P, T, lista_componentes, dados);
        fV_calc = resul['fv_flash']
        res_fV = f_V - fV_calc
        return res_fV
    sol = root(f_res_fV, T_guest, method='lm', 
               args=(f_V, z, P, lista_componentes, dados), tol=1.0e-8)
    T_flash = sol.x[0]
    return T_flash


def f_gera_mod_locais_elv(dados_elv):
    ''' Modelo locais do diagrama entalpia composição para usar no método
            McCabe-Thiele
            Entrada:
            dados_elv = dataframe do pandas com os dados de:
                T = temperatura em K
                x1 = composição de equilíbrio na fase líquida (liquido saturado)
                y1 = composição de equilíbrio na fase vapor (vapor saturado)
            Saidas:
            modelos = lista com os objetos dos respecitovos modelos na ordem:
                      mod_y_x, mod_x_y
            r2_modelos = lista com os valores dos coeficientes de determinação dos
                         modelos estimados
    '''
    # Modelo local para y1 = f(x1) - 3º grau
    x_regr = dados_elv['x1'].to_numpy().reshape(-1,1)
    y_regr = dados_elv['y1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_y_x = LinearRegression()
    mod_y_x.fit(X_poli_3g, y_regr)
    r2_y_x = r2_score(y_regr, mod_y_x.predict(X_poli_3g))
    # Modelo local para x1 = f(y1) - 3º grau
    x_regr = dados_elv['y1'].to_numpy().reshape(-1,1)
    y_regr = dados_elv['x1'].to_numpy()
    polinomio_3g = PolynomialFeatures(degree = 3)
    X_poli_3g = polinomio_3g.fit_transform(x_regr)
    mod_x_y = LinearRegression()
    mod_x_y.fit(X_poli_3g, y_regr)
    r2_x_y = r2_score(y_regr, mod_x_y.predict(X_poli_3g))
    #
    r2_modelos = [r2_y_x, r2_x_y]
    modelos = [mod_y_x, mod_x_y]
    #
    return (modelos, r2_modelos)

def f_uso_mod_loc_elv(x_var_ind, nome_var_dep, modelos):
    '''Função para usar os modelos locais estimados com: f_gera_mod_locais_ps
        Entradas:
        x_var_ind = valor da variável independente para o ual deseja calcular
                    a variável dependente
        nome_var_dep = string com o nome da variável dependente, podendo ser os
                    seguintes: 'y1' e 'x1'
        modelos = listas dos modelos estimados anteriormente
        Saidas:
        resp = valor calculado para a variável dependente
    '''
    #
    polinomio_2g = PolynomialFeatures(degree = 2)
    polinomio_3g = PolynomialFeatures(degree = 3)
    #
    resp = 0.0
    #
    if (nome_var_dep == 'y1'):
        mod = modelos[0]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    elif (nome_var_dep == 'x1'):
        mod = modelos[1]
        resp = mod.predict(polinomio_3g.fit_transform(np.array([x_var_ind]).reshape(-1,1)))[0]
    #
    return resp
