# Proyecto: Red Semántica de IA
# Autor: Henry Morales carne 2200304
# Asistencia: Elaborado con apoyo de herramientas de IA (ChatGPT, GPT-5 Thinking)


# --- arranque  para windows/utf-8 ---
import sys, os
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if os.name == "nt":
        import ctypes
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
except Exception:
    pass

import re                  # Expresiones regulares: buscar conceptos con patrones (regex).
import unicodedata         # Normalizar texto Unicode: quitar tildes/diacríticos y uniformar cadenas.
import collections         # Contenedores útiles (p.ej., deque) para BFS y manejo de niveles.
import textwrap            # Partir textos largos en varias líneas para etiquetas de nodos.
import networkx as nx      # Construir/manipular grafos dirigidos, layouts y caminos (shortest_path).
import matplotlib.pyplot as plt  # Dibujar y guardar el grafo en 2D (figuras PNG, estilos, etc.).
# fuente arial
import matplotlib                   # Núcleo de Matplotlib: config global (rcParams), backends y estilos.
from matplotlib import font_manager # Utilidades para gestionar y localizar fuentes (p.ej., buscar "Arial").
from matplotlib.patches import Patch # Formas/patches; usamos Patch para crear recuadros de color en la leyenda.


# --------------------------------------------------
# Manejo de Fuente, tipo letra 
# --------------------------------------------------
def _usar_arial():
    try:
        font_manager.findfont("arial", fallback_to_default=False)
        matplotlib.rcParams["font.family"] = "arial"
        matplotlib.rcParams["font.sans-serif"] = ["arial", "dejavu sans", "liberation sans"]
    except Exception:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["dejavu sans", "liberation sans", "arial"]

_usar_arial()

# --------------------------------------------------
# Utilidades de texto
# --------------------------------------------------
def _normalize(s) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

def _wrap_dos_lineas(s: str, width: int = 16) -> str:
    s = s.strip()
    if len(s) <= width:
        return s
    partes = textwrap.wrap(s, width=width)
    if not partes:
        return s
    if len(partes) == 1:
        return partes[0]
    return partes[0] + "\n" + " ".join(partes[1:])

# --------------------------------------------------
# Layout (intenta Graphviz; si no, fallback por capas)
# --------------------------------------------------
def _try_graphviz_layout(graf):
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        args = (
            "-Grankdir=TB "
            "-Granksep=1.6 "
            "-Gnodesep=1.2 "
            "-Gsplines=true "
            "-Goverlap=false "
            "-Gconcentrate=true"
        )
        return graphviz_layout(graf, prog="dot", args=args)
    except Exception:
        return None

def _niveles_por_bfs(graf, raiz: str):
    niveles = {n: None for n in graf.nodes}
    if raiz not in graf:
        return {n: 0 for n in graf.nodes}
    q = collections.deque([(raiz, 0)])
    niveles[raiz] = 0
    while q:
        n, d = q.popleft()
        for _, v in graf.out_edges(n):
            if niveles[v] is None or d + 1 < niveles[v]:
                niveles[v] = d + 1
                q.append((v, d + 1))
        for u, _ in graf.in_edges(n):
            if niveles[u] is None or d + 1 < niveles[u]:
                niveles[u] = d + 1
                q.append((u, d + 1))
    for n in graf.nodes:
        if niveles[n] is None:
            niveles[n] = 0
    return niveles

def _capas_por_bfs(graf, raiz):
    niveles = _niveles_por_bfs(graf, raiz)
    capas = {}
    for n, d in niveles.items():
        capas.setdefault(d, []).append(n)
    return capas

def _auto_figsize(graf, raiz, base=(16, 9), x_per_nodo=1.0, y_per_capa=1.3):
    capas = _capas_por_bfs(graf, raiz)
    if not capas:
        return base
    max_ancho = max(len(nodos) for nodos in capas.values())
    n_capas = len(capas)
    w = max(base[0], max_ancho * x_per_nodo)
    h = max(base[1], n_capas * y_per_capa)
    return (w, h)

def _posicion_chunked(graf, raiz="inteligencia artificial", max_por_fila=10):
    niveles = _niveles_por_bfs(graf, raiz)
    cont = {}
    layer_map = {}
    for n, lvl in niveles.items():
        idx = cont.get(lvl, 0)
        sub = idx // max_por_fila
        cont[lvl] = idx + 1
        layer_map[n] = lvl * 10 + sub
    g2 = graf.copy()
    nx.set_node_attributes(g2, layer_map, name="layer")
    return nx.multipartite_layout(g2, subset_key="layer", align="horizontal", scale=2.2)

def _posicion(graf, jerarquico=True, raiz="inteligencia artificial"):
    if jerarquico:
        pos = _try_graphviz_layout(graf)
        if pos is not None:
            return pos
        capas = _capas_por_bfs(graf, raiz)
        if capas and max(len(n) for n in capas.values()) > 10:
            return _posicion_chunked(graf, raiz, max_por_fila=10)
        niveles = _niveles_por_bfs(graf, raiz)
        nx.set_node_attributes(graf, niveles, name="layer")
        return nx.multipartite_layout(graf, subset_key="layer", align="horizontal", scale=2.0)
    return nx.spring_layout(graf, seed=42, k=0.9)

# --------------------------------------------------
# Construir red semántica mediante grafo dirigido  
# g.add_edge(nodo origen, nodo destino, label="etiqueta")
# --------------------------------------------------
g = nx.DiGraph()

# áreas
g.add_edge("inteligencia artificial", "aprendizaje automático", label="incluye")
g.add_edge("inteligencia artificial", "procesamiento de lenguaje natural", label="incluye")
g.add_edge("inteligencia artificial", "visión por computadora", label="incluye")

# enfoques
g.add_edge("aprendizaje automático", "aprendizaje supervisado", label="se divide en")
g.add_edge("aprendizaje automático", "aprendizaje no supervisado", label="se divide en")
g.add_edge("aprendizaje automático", "aprendizaje por refuerzo", label="se divide en")

# técnicas base
g.add_edge("aprendizaje supervisado", "clasificación", label="utiliza técnica")
g.add_edge("aprendizaje supervisado", "regresión", label="utiliza técnica")
g.add_edge("aprendizaje no supervisado", "agrupamiento", label="utiliza técnica")
g.add_edge("aprendizaje no supervisado", "reducción de dimensionalidad", label="utiliza técnica")

# algoritmos (refuerzo)
g.add_edge("aprendizaje por refuerzo", "q-learning", label="algoritmo")
g.add_edge("aprendizaje por refuerzo", "deep q-network", label="algoritmo")

# aplicaciones
g.add_edge("procesamiento de lenguaje natural", "análisis de sentimientos", label="aplicación")
g.add_edge("procesamiento de lenguaje natural", "traducción automática", label="aplicación")
g.add_edge("visión por computadora", "detección de objetos", label="aplicación")
g.add_edge("visión por computadora", "reconocimiento facial", label="aplicación")

# --- niveles extra ---
g.add_edge("aprendizaje automático", "aprendizaje profundo", label="se divide en")
g.add_edge("aprendizaje profundo", "redes neuronales", label="utiliza técnica")
g.add_edge("redes neuronales", "transformers", label="arquitectura")
g.add_edge("transformers", "bert", label="modelo")
g.add_edge("transformers", "gpt", label="modelo")

# visión por computador
g.add_edge("visión por computadora", "cnn", label="técnica")
g.add_edge("cnn", "resnet", label="arquitectura")
g.add_edge("resnet", "resnet-50", label="variante")
g.add_edge("resnet", "resnet-101", label="variante")

# supervisado
g.add_edge("clasificación", "svm", label="algoritmo")
g.add_edge("clasificación", "k-nn", label="algoritmo")
g.add_edge("regresión", "regresión lineal", label="algoritmo")

# no supervisado
g.add_edge("agrupamiento", "k-means", label="algoritmo")
g.add_edge("reducción de dimensionalidad", "pca", label="técnica")

# índice normalizado (para búsquedas/regex insensible a acentos)
index_normalizado = {node: _normalize(node) for node in g.nodes}

# --------------------------------------------------
# Categorías y colores 
# --------------------------------------------------
# ---------- categorías revisadas ----------
areas = {
    "inteligencia artificial",
    "aprendizaje automático",
    "procesamiento de lenguaje natural",
    "visión por computadora",
}

enfoques = {"aprendizaje supervisado", "aprendizaje no supervisado", "aprendizaje por refuerzo", "aprendizaje profundo"}

# TAREAS (tipos de problema)
tareas = {"clasificación", "regresión", "agrupamiento", "reducción de dimensionalidad"}

# MÉTODOS / ALGORITMOS
metodos = {
    "svm", "k-nn", "regresión lineal", "k-means", "pca",
    "q-learning", "deep q-network"
}

# ARQUITECTURAS (DL)
arquitecturas = {"redes neuronales", "cnn", "transformers", "resnet"}

# MODELOS concretos / variantes
modelos = {"bert", "gpt", "resnet-50", "resnet-101"}

# APLICACIONES
aplicaciones = {"análisis de sentimientos", "traducción automática", "detección de objetos", "reconocimiento facial"}

def _color_nodo(n):
    if n in areas:         return "#2b8cbe"   # áreas
    if n in enfoques:      return "#7bccc4"   # enfoques
    if n in tareas:        return "#ffb703"   # tareas
    if n in metodos:       return "#fb8500"   # métodos/algoritmos
    if n in arquitecturas: return "#bc5090"   # arquitecturas
    if n in modelos:       return "#8338ec"   # modelos
    if n in aplicaciones:  return "#adb5bd"   # aplicaciones
    return "#8ecae6"                         # por defecto

CATEGORIES = [
    ("áreas",             areas,         "#2b8cbe"),
    ("enfoques",          enfoques,      "#7bccc4"),
    ("tareas",            tareas,        "#ffb703"),
    ("métodos/algoritmos",metodos,       "#fb8500"),
    ("arquitecturas",     arquitecturas, "#bc5090"),
    ("modelos",           modelos,       "#8338ec"),
    ("aplicaciones",      aplicaciones,  "#adb5bd"),
]


# --------------------------------------------------
# Dibujo 2D 
# --------------------------------------------------
def dibujar_red(
    subgrafo,
    nombre_archivo,
    titulo="red semántica - ia",
    jerarquico=True,
    font_node=8,
    font_edge=7,
    node_size=1600,
    label_width=16,
    show_legend=True,
    show_edge_labels=True,
    edge_color="#888888",
    highlight_nodes=None,         
    dpi=200
):
    pos = _posicion(subgrafo, jerarquico=jerarquico, raiz="inteligencia artificial")
    labels_nodos = {n: _wrap_dos_lineas(n, width=label_width) for n in subgrafo.nodes}
    colores = [_color_nodo(n) for n in subgrafo.nodes]

    plt.figure(figsize=_auto_figsize(subgrafo, "inteligencia artificial", base=(16, 9)))

    nx.draw(
        subgrafo, pos,
        with_labels=False,
        node_color=colores,
        node_size=node_size,
        edgecolors="#1d3557",
        linewidths=0.6,
        arrowsize=14,
        width=0.9,
        edge_color=edge_color,
        alpha=0.98
    )
    nx.draw_networkx_labels(
        subgrafo, pos,
        labels=labels_nodos,
        font_size=font_node,
        font_weight="bold",
        font_family="arial"
    )

    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(subgrafo, "label")
        nx.draw_networkx_edge_labels(
            subgrafo, pos,
            edge_labels=edge_labels,
            font_size=font_edge,
            label_pos=0.5,
            font_family="arial",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
        )

    # Resalta nodos destino (overlay)
    if highlight_nodes:
        highlight_nodes = [n for n in highlight_nodes if n in subgrafo.nodes]
        if highlight_nodes:
            nx.draw_networkx_nodes(
                subgrafo, pos,
                nodelist=highlight_nodes,
                node_size=int(node_size * 1.08),
                node_color="none",
                edgecolors="#d90429",    # rojo
                linewidths=2.2
            )

    if show_legend:
        handles = []
        for nombre, conjunto, color in CATEGORIES:
            if any(n in conjunto for n in subgrafo.nodes):
                handles.append(Patch(facecolor=color, edgecolor="#1d3557", label=nombre))
        if handles:
            plt.legend(handles=handles, title="categorías", loc="lower right", frameon=True, framealpha=0.9)

    plt.title(titulo, fontsize=13, fontweight="bold", fontfamily="arial")
    plt.margins(0.05)
    plt.savefig(nombre_archivo, dpi=dpi, bbox_inches="tight")
    plt.show()

# --------------------------------------------------
# Búsquedas
# --------------------------------------------------
def _coincidencias_regex(patron: str):
    patron_norm = _normalize(patron)
    try:
        rx = re.compile(patron_norm)
    except re.error as e:
        print(f"[!] expresión regular inválida: {e}", flush=True)
        return []
    return [n for n, norm in index_normalizado.items() if rx.search(norm)]

def buscar_regex(patron: str):
    """Subgrafo de coincidencias + vecinos inmediatos (búsqueda normal)."""
    coincidencias = _coincidencias_regex(patron)
    if not coincidencias:
        print(f"[!] no encontré conceptos que coincidan con: '{patron}'", flush=True)
        return None

    nodos_sub = set(coincidencias)
    for n in coincidencias:
        nodos_sub.update(v for _, v in g.out_edges(n))
        nodos_sub.update(u for u, _ in g.in_edges(n))

    sub = g.subgraph(nodos_sub).copy()
    titulo = f"relaciones para patrón: '{patron}'"
    archivo = f"busqueda_{_normalize(patron)}.png"
    dibujar_red(
        sub, archivo, titulo=titulo, jerarquico=True,
        font_node=8, font_edge=7, node_size=1650, label_width=16,
        show_legend=True, show_edge_labels=True, edge_color="#666666",
        highlight_nodes=coincidencias
    )
    return coincidencias

def _agregar_arista_con_label(h, u, v):
    """Añade a h la arista u→v usando la dirección/label existente en g."""
    if g.has_edge(u, v):
        h.add_edge(u, v, label=g[u][v].get("label", ""))
    elif g.has_edge(v, u):
        h.add_edge(v, u, label=g[v][u].get("label", ""))

def buscar_ruta_regex(patron: str, raiz="inteligencia artificial"):
    #Dibuja el camino  desde la raíz hasta cada coincidencia del patrón.
 
    coincidencias = _coincidencias_regex(patron)
    if not coincidencias:
        print(f"[!] no encontré conceptos que coincidan con: '{patron}'", flush=True)
        return None

    und = g.to_undirected()
    h = nx.DiGraph()
    # copia colores por categoría con solo nodos del camino
    for destino in coincidencias:
        try:
            path = nx.shortest_path(und, raiz, destino)
        except nx.NetworkXNoPath:
            print(f"[!] no hay ruta entre '{raiz}' y '{destino}'.")
            continue
        # Añadimos nodos del camino
        for n in path:
            h.add_node(n)
        # Añadimos aristas del camino preservando sentido/label original
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            _agregar_arista_con_label(h, u, v)

    if len(h) == 0:
        print("[!] no se pudo construir un camino con ese patrón.")
        return None

    titulo = f"camino(s) desde '{raiz}' hacia patrón: '{patron}'"
    archivo = f"ruta_{_normalize(patron)}.png"
    dibujar_red(
        h, archivo, titulo=titulo, jerarquico=True,
        font_node=9, font_edge=8, node_size=1700, label_width=18,
        show_legend=True, show_edge_labels=True, edge_color="#333333",
        highlight_nodes=coincidencias
    )
    return coincidencias

# --------------------------------------------------
# Menú
# --------------------------------------------------
def mostrar_menu():
    print("\n--- menú red semántica ia ---", flush=True)
    print("1. ver gráfico general")
    print("2. buscar un concepto con subgrafo con vecinos")
    print("3. buscar ruta completa de un concepto")
    print("4. salir")
def loop():
    while True:
        mostrar_menu()
        entrada = input("seleccione una opción o comando: ").strip()
        if not entrada:
            continue
        partes = entrada.split(maxsplit=1)
        cmd = partes[0].lower()

        if cmd in {"1", "ver", "general"}:
            dibujar_red(
                g, "grafico_general.png",
                titulo="red semántica - inteligencia artificial",
                jerarquico=True,
                show_edge_labels=True
            )
        elif cmd in {"2", "buscar"}:
            patron = partes[1] if len(partes) > 1 else input("ingrese concepto o regex: ").strip()
            encontrados = buscar_regex(patron)
            if encontrados:
                print("coincidencias:", ", ".join(sorted(encontrados)), flush=True)
        elif cmd in {"3", "ruta"}:
            patron = partes[1] if len(partes) > 1 else input("ingrese concepto o regex: ").strip()
            encontrados = buscar_ruta_regex(patron)
            if encontrados:
                print("destinos:", ", ".join(sorted(encontrados)), flush=True)
        elif cmd in {"4", "salir", "exit", "q"}:
            print("saliendo...", flush=True)
            break
        else:
            print("opción/comando no reconocido.", flush=True)

if __name__ == "__main__":
    print("iniciando menú...", flush=True)
    loop()
