from bs4 import BeautifulSoup
import csv

html_content = """<div class="ue-l-common-page"><div class="ue-l-common-page__inner"><div class="recursos-deportivos">
<div class="pag-octavos">
<div class="sub-tabs-content">
<ul class="sub-tab-list">
<li><a class="tab-octavos" href="octavos.html">Octavos de final</a></li>
<li><a class="tab-cuartos" href="cuartos.html">Cuartos de final</a></li>
<li><a class="tab-semifinales" href="semifinales.html">Semifinales</a></li>
<li><a class="tab-final" href="final.html">Final</a></li>
 </ul>
</div>
</div>
<style type="text/css" media="screen">.recursos-deportivos .ronda-ida caption::after{content:" - Octavos de final (Ida)"}.recursos-deportivos .ronda-vuelta caption::after{content:" - Octavos de final (Vuelta)"}</style>
<div class="contenedorCalendarioInt" id="contenedorCalendarioInt" onclick="cambiaEquipo(event)">
<div class="ronda-ida">
<div class="cal-agendas calendario">
<div class="jornada datos-jornada">
<table class="jor agendas" cellpadding="0" cellspacing="0" summary="Todos los resultados de la jornada">
<caption>Jornada 7</caption>
<thead>
<tr>
<th scope="col">Equipo local</th>
<th scope="col">Resultado</th>
<th scope="col">Equipo visitante</th>
</tr>
</thead>
<tbody>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/569.png" alt="FC Copenhague">
</figure>
<span class="equipo_t569">FC Copenhague</span>
</td>
<td class="resultado"><span class="resultado-partido">1-3</span></td>
<td class="visitante">
<span class="equipo_t43">M. City</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/43.png" alt="M. City">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/6339.png" alt="Leipzig">
</figure>
<span class="equipo_t6339">Leipzig</span>
</td>
<td class="resultado"><span class="resultado-partido">0-1</span></td>
<td class="visitante">
<span class="equipo_t186">Real Madrid</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/186.png" alt="Real Madrid">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/129.png" alt="Lazio">
</figure>
<span class="equipo_t129">Lazio</span>
</td>
<td class="resultado"><span class="resultado-partido">1-0</span></td>
<td class="visitante">
<span class="equipo_t156">Bayern Múnich</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/156.png" alt="Bayern Múnich">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/149.png" alt="PSG">
</figure>
<span class="equipo_t149">PSG</span>
</td>
<td class="resultado"><span class="resultado-partido">2-0</span></td>
<td class="visitante">
<span class="equipo_t188">R. Sociedad</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/188.png" alt="R. Sociedad">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/127.png" alt="Inter Milán">
</figure>
<span class="equipo_t127">Inter Milán</span>
</td>
<td class="resultado"><span class="resultado-partido">1-0</span></td>
<td class="visitante">
<span class="equipo_t175">Atlético</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/175.png" alt="Atlético">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/204.png" alt="PSV">
</figure>
<span class="equipo_t204">PSV</span>
</td>
<td class="resultado"><span class="resultado-partido">1-1</span></td>
<td class="visitante">
<span class="equipo_t157">Borussia Dortmund</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/157.png" alt="Borussia Dortmund">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/201.png" alt="Oporto">
</figure>
<span class="equipo_t201">Oporto</span>
</td>
<td class="resultado"><span class="resultado-partido">1-0</span></td>
<td class="visitante">
<span class="equipo_t3">Arsenal</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/3.png" alt="Arsenal">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/459.png" alt="Nápoles">
</figure>
<span class="equipo_t459">Nápoles</span>
</td>
<td class="resultado"><span class="resultado-partido">1-1</span></td>
<td class="visitante">
<span class="equipo_t178">Barcelona</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/178.png" alt="Barcelona">
</figure>
</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
<div class="ronda-vuelta">
<div class="cal-agendas calendario">
<div class="jornada datos-jornada">
<table class="jor agendas" cellpadding="0" cellspacing="0" summary="Todos los resultados de la jornada">
<caption>Jornada 8</caption>
<thead>
<tr>
<th scope="col">Equipo local</th>
<th scope="col">Resultado</th>
<th scope="col">Equipo visitante</th>
</tr>
</thead>
<tbody>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/156.png" alt="Bayern Múnich">
</figure>
<span class="equipo_t156">Bayern Múnich</span>
</td>
<td class="resultado"><span class="resultado-partido">3-0</span></td>
<td class="visitante">
<span class="equipo_t129">Lazio</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/129.png" alt="Lazio">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/188.png" alt="R. Sociedad">
</figure>
<span class="equipo_t188">R. Sociedad</span>
</td>
<td class="resultado"><span class="resultado-partido">1-2</span></td>
<td class="visitante">
<span class="equipo_t149">PSG</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/149.png" alt="PSG">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/43.png" alt="M. City">
</figure>
<span class="equipo_t43">M. City</span>
</td>
<td class="resultado"><span class="resultado-partido">3-1</span></td>
<td class="visitante">
<span class="equipo_t569">FC Copenhague</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/569.png" alt="FC Copenhague">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/186.png" alt="Real Madrid">
</figure>
<span class="equipo_t186">Real Madrid</span>
</td>
<td class="resultado"><span class="resultado-partido">1-1</span></td>
<td class="visitante">
<span class="equipo_t6339">Leipzig</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/6339.png" alt="Leipzig">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/3.png" alt="Arsenal">
</figure>
<span class="equipo_t3">Arsenal</span>
</td>
<td class="resultado"><span class="resultado-partido">1-0</span></td>
<td class="visitante">
<span class="equipo_t201">Oporto</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/201.png" alt="Oporto">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/178.png" alt="Barcelona">
</figure>
<span class="equipo_t178">Barcelona</span>
</td>
<td class="resultado"><span class="resultado-partido">3-1</span></td>
<td class="visitante">
<span class="equipo_t459">Nápoles</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/459.png" alt="Nápoles">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/175.png" alt="Atlético">
</figure>
<span class="equipo_t175">Atlético</span>
</td>
<td class="resultado"><span class="resultado-partido">2-1</span></td>
<td class="visitante">
<span class="equipo_t127">Inter Milán</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/127.png" alt="Inter Milán">
</figure>
</td>
</tr>
<tr>
<td class="local">
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/157.png" alt="Borussia Dortmund">
</figure>
<span class="equipo_t157">Borussia Dortmund</span>
</td>
<td class="resultado"><span class="resultado-partido">2-0</span></td>
<td class="visitante">
<span class="equipo_t204">PSV</span>
<figure>
<img src="https://e00-marca.uecdn.es/assets/sports/logos/football/png/72x72/204.png" alt="PSV">
</figure>
</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
<script type="text/javascript">function getElementsByClass(searchClass,node,tag){var classElements=new Array();if(node==null)
node=document;if(tag==null)
tag='*';var els=node.getElementsByTagName(tag);var elsLen=els.length;var pattern=new RegExp("(^|\\s)"+searchClass+"(\\s|$)");for(var i=0,j=0;i<elsLen;i++){if(pattern.test(els[i].className)){classElements[j]=els[i];j++;}}
return classElements;}
function cambiaEquipo(e){e=e||window.event;var obj=e.target||e.srcElement;if(obj.className.indexOf('equipo_')==-1)return false;var node=document.getElementById('contenedorCalendarioInt');var equipos=getElementsByClass('seleccionado',node,'span');for(var i=0;i<equipos.length;++i){equipos[i].className=equipos[i].className.replace('seleccionado','');}
equipos=getElementsByClass(obj.className,node,'span');for(var i=0;i<equipos.length;++i){equipos[i].className+=' seleccionado';}}</script>
</div>
</div></div>"""  # Asegúrate de colocar tu HTML entre las comillas

soup = BeautifulSoup(html_content, 'html.parser')

# Encuentra todas las tablas dentro del contenido
tablas = soup.find_all('table', class_='jor agendas')

# Lista para guardar los datos extraídos
datos_partidos = []

# Procesa cada tabla
for tabla in tablas:
    rondas = tabla.find('caption').text.strip()  # Extrae la ronda de la tabla
    filas = tabla.find_all('tr')[1:]  # Ignora el encabezado
    for fila in filas:
        equipo_local = fila.find('td', class_='local').text.strip()
        resultado = fila.find('td', class_='resultado').text.strip()
        equipo_visitante = fila.find('td', class_='visitante').text.strip()

        datos_partidos.append({
            'Ronda': rondas,
            'Equipo Local': equipo_local,
            'Resultado': resultado,
            'Equipo Visitante': equipo_visitante
        })

# Guarda los datos en un archivo CSV
with open('resultados_partidos.csv', 'w', newline='', encoding='utf-8') as archivo:
    campos = ['Ronda', 'Equipo Local', 'Resultado', 'Equipo Visitante']
    writer = csv.DictWriter(archivo, fieldnames=campos)
    writer.writeheader()
    writer.writerows(datos_partidos)

print("Los datos han sido guardados en 'resultados_partidos.csv'.")
