# Fixa duas listas de pontos, A = [a1,...,a10] e B = [b1,...,b10] sobre uma esfera e as geodésicas que ligam cada a_i a b_i.

# Então, coleta uma amostra aleatória de pontos sobre a esfera, adiciona A e B à lista de pontos e faz o seguinte:
# cria um grafo a partir de todos esses pontos usando o KNN e calcula o caminho mínimo (Dijkstra) que liga a_i a b_i no grafo,
# para cada i = 1,...,10.

# Esses caminhos mínimmos são aproximações para as geodésicas, e seus erros são calculados pela diferença
# de comprimento entre cada caminho e a respectiva geodésica exata (as geodésicas exatas e seus comprimentos foram coletadas e 
# estão fixadas no início deste código).

# Após isso, calcula a média dos erros absolutos e relativos e guarda o valor.

# Repete o processo de criação dos caminhos e comparação com as geodésicas exatas para diferentes 
# amostras (mantendo apenas A e B) com diferentes quantidades de pontos, e plota um gráfico mostrando as médias dos erros absolutos 
# e relativos para cada quantidade de amostra


from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx





## GEODÉSICAS FIXADAS E SEUS COMPRIMENTOS
geodesica_exata1 = np.array([[0.7711239252813405, 0.08140853602246934, 0.6314590581513388], [0.7857583148979481, 0.06833187279938001, 0.6176515525939266], [0.8041757749499547, 0.04778140539483121, 0.5930542910208506], [0.8083224044530176, 0.04372020592955049, 0.5885140933958157], [0.8225932350019611, 0.027230937990282374, 0.5684570294477743], [0.8347907507993737, 0.012772269783654477, 0.5506700643122505], [0.8449149518452548, 0.0003442013096669472, 0.5351531979892442], [0.8529658381396042, -0.010053267431680252, 0.521906430478755], [0.8610167244339539, -0.020450736173027423, 0.508659662968266], [0.8669942959767719, -0.028817605181734277, 0.4976829942702944], [0.8729718675195902, -0.03718447419044109, 0.48670632557232285], [0.8828536958536953, -0.051887612475214534, 0.46702308698889694], [0.8927355241878001, -0.06659075075998785, 0.44733984840547125], [0.8984707230188428, -0.07723268957948073, 0.43219680744708044], [0.9081101786411716, -0.09421089767503996, 0.40834719660323515], [0.9138453774722142, -0.10485283649453267, 0.3932041556448444], [0.9214115183430116, -0.1198004448574517, 0.37162464361351644], [0.9232424603827671, -0.12410611440087782, 0.3651881725405799], [0.9250734024225225, -0.12841178394430408, 0.3587517014676429], [0.9305662285417888, -0.14132879257458275, 0.3394422882488327], [0.9358166819492792, -0.15652087101564713, 0.3159665027696033], [0.9410671353567697, -0.1717129494567117, 0.29249071729037357], [0.9460752160524841, -0.18918009770856203, 0.2648485595507247], [0.9476637853804636, -0.19576083706277425, 0.25424571621736847], [0.9505985513246464, -0.2111973855819842, 0.228873657290237], [0.9519447479408503, -0.22005319474698226, 0.21410444169646153], [0.9532909445570539, -0.2289090039119802, 0.19933522610268622], [0.9541523957497056, -0.24231495269854997, 0.17623326598807204], [0.9547714742305811, -0.25799597129590546, 0.14896493361303853], [0.9553905527114569, -0.2736769898932611, 0.12169660123800513], [0.9551481799996809, -0.27595205970404685, 0.11753022897758608], [0.9549058072879051, -0.27822712951483297, 0.11336385671716637], [0.9539363164408012, -0.28732740875797624, 0.0966983676754894], [0.9536939437290253, -0.28960247856876237, 0.09253199541506968], [0.9529668255936975, -0.2964276880011199, 0.08003287863381181], [0.9491969018597343, -0.31032257694398063, 0.053138371623393965], [0.9489545291479586, -0.3125976467547667, 0.04897199936297447], [0.9451846054139954, -0.3264925356976274, 0.022077492352556627], [0.9416570543918086, -0.3381123548297023, -0.0006506423974418363], [0.9386142487931732, -0.3451820343402055, -0.015046032626601813], [0.9355714431945382, -0.3522517138507089, -0.029441422855762123], [0.9325286375959032, -0.35932139336121205, -0.04383681308492182], [0.9266853991104087, -0.37118568257143253, -0.0684612212828224], [0.9210845333366899, -0.38077490197086705, -0.08891925722030358], [0.9129256073878882, -0.39288366125923313, -0.11543993886610593], [0.9050090541508626, -0.4027173507368134, -0.13779424825148917], [0.8945344407387538, -0.415070580103325, -0.1662112033451934], [0.8840598273266447, -0.4274238094698366, -0.1946281584388978], [0.8815017671515615, -0.42994334935876827, -0.20069080414721935], [0.868953838987921, -0.4402659789926393, -0.22683766042840597], [0.8564059108242806, -0.45058860862651073, -0.2529845167095931], [0.8464160428357235, -0.45839169837145055, -0.2730687272824587], [0.834110487383859, -0.46643925819453597, -0.29504921130322626], [0.8218049319319944, -0.47448681801762127, -0.3170296953239937], [0.8120574366552133, -0.4800148379517753, -0.33294753363644014], [0.8071836890168229, -0.4827788479188524, -0.3409064527926635], [0.7902467586383435, -0.49131534789822884, -0.36667948370923503], [0.7781835758982552, -0.4970878379105284, -0.3844935954695833], [0.7709941407965571, -0.5000963179557509, -0.39434878807370816], [0.7566152705931612, -0.5061132780461961, -0.41405917328195846], [0.7494258354914634, -0.5091217580914185, -0.4239143658860834], [0.7327312778247601, -0.5153831882600092, -0.44552102454223563], [0.7114053452314424, -0.522133558584891, -0.4709202300941916], [0.6995845352031297, -0.5256309787864045, -0.4845679695941204], [0.6854480377115098, -0.529372869066064, -0.5001119825419515], [0.6713115402198898, -0.5331147593457229, -0.5156559954897821], [0.6502279803383478, -0.5375900598598187, -0.5368888287813186], [0.6245130455301915, -0.5425543005302059, -0.5619142089686593], [0.6034294856486494, -0.5470296010443016, -0.5831470422601959], [0.601113798185342, -0.5472740711224472, -0.5850433157080978], [0.5825882984788835, -0.5492298317476116, -0.6002135032913131], [0.5663784862357322, -0.5509411222946304, -0.6134874174266265], [0.550168673992581, -0.5526524128416493, -0.6267613315619399], [0.5270117993595075, -0.5550971136231049, -0.6457240660409592], [0.4971502350482884, -0.5560001548282113, -0.6662092486032647], [0.4925188601216737, -0.5564890949845025, -0.6700017954990686], [0.4672886707370688, -0.5569031960333175, -0.6866944311655702], [0.4420584813524642, -0.5573172970821328, -0.7033870668320721], [0.4237753543577816, -0.5569979878965114, -0.7143908821548683], [0.40549222736309887, -0.5566786787108898, -0.725394697477664], [0.3895247878317236, -0.556114899447123, -0.7345022393525583], [0.37355734830034804, -0.5555511201833557, -0.7436097812274522], [0.35990559623228013, -0.5547428708414434, -0.7508210496544445], [0.3349177795594514, -0.5528819020794726, -0.7633473130605268], [0.32358171495469085, -0.5518291826594146, -0.7686623080396171], [0.30090958574516924, -0.5497237438192983, -0.7792922979977976], [0.2895735211404087, -0.5486710243992402, -0.784607292976888], [0.26921707939419454, -0.5463211154809784, -0.7933410094871667], [0.23984026050652713, -0.542674017064513, -0.8054934475286337], [0.21277912908216684, -0.5387824485699019, -0.8157496121221983], [0.18803368512111424, -0.5346464099971454, -0.8241095032678615], [0.16328824116006135, -0.5305103714243888, -0.8324693944135247], [0.1408584846623161, -0.5261298627734868, -0.8389330121112858], [0.11172403848642432, -0.5202076945462354, -0.8469190778923334], [0.08258959231053303, -0.5142855263189843, -0.8549051436733811], [0.0758849026323869, -0.5127438667426348, -0.8564275917566673], [0.04467714170496401, -0.5047910987827432, -0.8621435587251975], [0.013469380777541118, -0.4968383308228515, -0.8678595256937276], [-0.011033690471735913, -0.490427222439309, -0.8720530445789714], [-0.039925763935851455, -0.48222998440127174, -0.8758727380995996], [-0.062113147721820805, -0.47557440593958394, -0.8781699835369416], [-0.08868953372262894, -0.46713269782340117, -0.8800934036096679], [-0.10417222783045232, -0.4620187789380625, -0.8808682009637234], [-0.1196549219382761, -0.4569048600527235, -0.8816429983177785], [-0.13513761604609947, -0.4517909411673847, -0.882417795671834], [-0.15939831458360038, -0.44310476297305645, -0.8824449422966585], [-0.1880480153359399, -0.4326324551242335, -0.8820982635568675], [-0.1924370175507788, -0.4308463254697385, -0.8817244381922518], [-0.22547572051795695, -0.41858788796642094, -0.8810039340878455], [-0.22986472273279573, -0.41680175831192595, -0.8806301087232298], [-0.23425372494763436, -0.41501562865743136, -0.8802562833586145],[-0.26376702, -0.40238501, -0.87664888]])
comprimento_geodesica1 = 2.482589784849599

geodesica_exata2 = np.array([[0, 0, 1], [0.0, 0.021021021021021102, 1.0015015015015014], [0.0, 0.057057057057057214, 0.9984984984984986], [0.0, 0.09909909909909898, 0.9954954954954953], [0.0, 0.1231231231231229, 0.9924924924924925], [0.0, 0.1651651651651651, 0.9864864864864864], [0.0, 0.18318318318318294, 0.9834834834834836], [0.0, 0.22522522522522515, 0.9744744744744747], [0.0, 0.26126126126126126, 0.9654654654654653], [0.0, 0.30330330330330346, 0.9534534534534536], [0.0, 0.3213213213213213, 0.9474474474474475], [0.0, 0.36936936936936915, 0.9294294294294292], [0.0, 0.40540540540540526, 0.9144144144144142], [0.0, 0.44744744744744747, 0.8963963963963963], [0.0, 0.48948948948948967, 0.8723723723723724], [0.0, 0.5255255255255253, 0.8513513513513513], [0.0, 0.5495495495495497, 0.8363363363363363], [0.0, 0.5795795795795797, 0.8153153153153152], [0.0, 0.6036036036036037, 0.7972972972972974], [0.0, 0.6456456456456454, 0.7642642642642641], [0.0, 0.6756756756756754, 0.7372372372372373], [0.0, 0.7117117117117115, 0.704204204204204], [0.0, 0.7417417417417416, 0.6711711711711712], [0.0, 0.7777777777777777, 0.629129129129129], [0.0, 0.8078078078078077, 0.5900900900900901], [0.0, 0.8318318318318316, 0.5570570570570572], [0.0, 0.8498498498498499, 0.5270270270270272], [0.0, 0.8678678678678677, 0.49699699699699695], [0.0, 0.885885885885886, 0.4639639639639639], [0.0, 0.9039039039039038, 0.427927927927928], [0.0, 0.9219219219219217, 0.38888888888888884], [0.0, 0.9279279279279278, 0.37387387387387383], [0.0, 0.945945945945946, 0.32582582582582575], [0.0, 0.9519519519519517, 0.3078078078078077], [0.0, 0.9639639639639639, 0.26876876876876876], [0.0, 0.96996996996997, 0.24474474474474484], [0.0, 0.9819819819819822, 0.19069069069069067], [0.0, 0.9879879879879878, 0.15465465465465456], [0.0, 0.9939939939939939, 0.10960960960960953], [0.0, 1.0, 0.06156156156156145], [0.0, 1.0, 0.04054054054054057], [0, 1, 0]])
comprimento_geodesica2 = 1.5716855502020577

geodesica_exata3 = np.array([[0.8775825618903728, 0.479425538604203, 0.0], [0.8675798561599113, 0.5008672449684564, 0.001868333890378638], [0.8464201866578108, 0.5333360075060629, 0.004925607529180082], [0.8273187712038615, 0.5620371612803632, 0.00764318409700356], [0.8123338638462148, 0.5832030975280527, 0.00968136652287118], [0.797348956488568, 0.604369033775742, 0.011719548948738803], [0.7756121579658789, 0.6316304712178977, 0.01443712551656228], [0.7559336134913409, 0.6551242998967479, 0.01681500501340784], [0.7471234682781478, 0.6649874098545201, 0.01783409622634164], [0.7227512866867184, 0.6908091309645311, 0.020551672794165143], [0.7071892503084831, 0.7067677421167702, 0.022250158149054823], [0.6848753227652052, 0.7288218544634752, 0.02462803764590036], [0.6781234316001631, 0.734917355657942, 0.02530743178785624], [0.671371540435121, 0.7410128568524087, 0.025986825929812125], [0.6578677581050365, 0.7532038592413418, 0.027345614213723862], [0.6329184474930188, 0.7738182552559022, 0.0297234937105694], [0.6147210280460436, 0.7883371500759964, 0.03142197906545908], [0.5965236085990682, 0.8028560448960904, 0.03312046442034876], [0.5689389149183106, 0.8220307245785063, 0.03549834391719432], [0.536660584120662, 0.843533296692083, 0.0382159204850178], [0.5111341444880556, 0.8589403676111932, 0.04025410291088542], [0.48091406773855844, 0.8766753309614645, 0.04263198240773098], [0.4715267935047762, 0.8813311158237864, 0.043311376549686864], [0.46683315638788553, 0.8836590082549471, 0.04365107362066478], [0.4433649708034303, 0.8952984704107518, 0.04534955897555446], [0.4245904223358659, 0.9046100401353956, 0.046708347259466224], [0.4198967852189752, 0.9069379325665563, 0.04704804433044414], [0.4152031481020839, 0.9092658249977172, 0.04738774140142208], [0.37971230521510685, 0.9241213556836989, 0.049765620898267644], [0.35360873656191205, 0.9343211015073589, 0.051464106253157324], [0.33219880502560806, 0.9421929548998578, 0.052822894537069065], [0.31548251060619503, 0.9477369158611955, 0.05384198575000286], [0.28674355888426, 0.9564969453527108, 0.05554047110489257], [0.26899798, 0.9614786 , 0.05655956]])
comprimento_geodesica3 = 0.7999811738589561

geodesica_exata4 = np.array([[0.18163563200134014, -0.5590169943749473, 0.8090169943749475], [0.16638605989236582, -0.549374936701286, 0.8209591925447765], [0.13824541059619064, -0.5290625669349728, 0.8376007484154722], [0.1101047613000154, -0.5087501971686592, 0.8542423042861674], [0.09231231085927372, -0.4954254424223994, 0.8637322215315066], [0.07451986041853204, -0.4821006876761395, 0.873222138776846], [0.06017680959626796, -0.4711051379365637, 0.8803281764803995], [0.03494010757021829, -0.45144324346409714, 0.892156372345721], [0.005708553577338099, -0.42777341427210835, 0.9039389672892398], [-0.012629349211755986, -0.41276992981301064, 0.9109994040709904], [-0.0458557551714662, -0.3850921659014999, 0.9227363980927065], [-0.07218336189422087, -0.36207281200335784, 0.9297056330308512], [-0.10250582058380558, -0.3350455233856939, 0.936629267047193], [-0.12538402768808232, -0.3143553744942362, 0.9412146224435523], [-0.1562519387260194, -0.28564935616373466, 0.9457087759963057], [-0.17967559817864848, -0.2632804775594394, 0.9478646509290767], [-0.20709410959810787, -0.23690366423562212, 0.9499749249400449], [-0.21907866549859853, -0.22487986007705552, 0.9498381221746357], [-0.22307351746542878, -0.22087192535753347, 0.949792521252833], [-0.2510374812332406, -0.1928163823208784, 0.9494733148002126], [-0.27555204538257444, -0.16709004429090774, 0.9467702288058064], [-0.296071757565078, -0.14537164098045946, 0.9441127437332035], [-0.31259661778075126, -0.12766117238953323, 0.9415008595824034], [-0.3251266260295943, -0.1139586385181291, 0.938934576353406], [-0.34619179056045013, -0.09056150549484321, 0.9338476108172148], [-0.37179725940648883, -0.06147770803919722, 0.9262855638956318], [-0.39340787628569707, -0.036401845303073554, 0.918769117895852], [-0.4064833368828925, -0.02102058171883181, 0.9137733542032662], [-0.42409910179527077, 4.734629776997901e-05, 0.9063025091252894], [-0.4462551710228315, 0.02680193874673148, 0.8963565826619209], [-0.46500968,  0.05137456,  0.8838137 ]])
comprimento_geodesica4 = 0.92549591588648

geodesica_exata5 = np.array([[0.47552825814757665, -0.3454915028125263, 0.8090169943749475], [0.491929954159511, -0.34497394843854945, 0.8014532993749492], [0.5263595176427253, -0.3415677269684782, 0.7789588240958358], [0.556306884092134, -0.33845759417960847, 0.7589724731686626], [0.5817720535077374, -0.3356435500719405, 0.7414942465934303], [0.5982728288557295, -0.3334216833266759, 0.7290322687220795], [0.6267921825179084, -0.3292740385173482, 0.706616437331318], [0.6388107608320949, -0.3273482604532851, 0.6966625838119074], [0.6703842987408491, -0.32186701494229775, 0.6693091476056171], [0.7050120208961794, -0.3150521687296507, 0.6370181065837975], [0.730675348983898, -0.3088294998794069, 0.6097433142658593], [0.7593928613181926, -0.3012732303275034, 0.5775309171323918], [0.7836281766186814, -0.2940130494568016, 0.5478266443508653], [0.7972729263922139, -0.2897161586706209, 0.5305057055523372], [0.8200802289054725, -0.28141846577946095, 0.49837195230722203], [0.8428875314187317, -0.27312077288830117, 0.46623819906210695], [0.8581584526516095, -0.26645276938000273, 0.4415501749844615], [0.8781097295904084, -0.25608005244792686, 0.4045574608121697], [0.895006822282632, -0.24704093621751058, 0.3725023514554067], [0.9135300864342011, -0.2356307072649766, 0.333080156819526], [0.925944982092619, -0.226887679715762, 0.30353317181470374], [0.9353056935044614, -0.2194782528682072, 0.2789237916254107], [0.9462925763756493, -0.20969771329853465, 0.2469473261570001], [0.9525991035409163, -0.20362188715263946, 0.22727555078323586], [0.9621579736248743, -0.19280383556250877, 0.19286960485123655], [0.9717168437088322, -0.18198578397237808, 0.1584636589192373], [0.9784196882183304, -0.16909270834133144, 0.11919875206006147], [0.9834963612684826, -0.1585707454324024, 0.08730093048000293], [0.988771192990751, -0.1446401577808978, 0.04560654315723883], [0.9907936817943277, -0.1354517955736284, 0.018646326392709245], [0.9930143292700205, -0.12285480862378342, -0.018110456114526163], [0.9923789511712527, -0.10818279763302228, -0.05972619954893821], [0.9915454144003704, -0.0969194113848369, -0.09154537724064477], [0.9892838648422573, -0.08461851311619362, -0.12579403539593925], [0.9841662897096842, -0.07024259080663411, -0.16490165447841082], [0.9776207017898813, -0.054829156476616646, -0.20643875402447065], [0.9708769551979628, -0.04282434688917489, -0.23817928782782483], [0.9612771830315846, -0.028744513260817124, -0.2747787825583557], [0.9557632905547802, -0.021185840436409198, -0.2942932701554153], [0.9488213852907463, -0.012589655591543174, -0.31623723821606375], [0.9404514672394819, -0.0029559587262192114, -0.34061068674030004], [0.9292255236137577, 0.008752762180020843, -0.3698430961917133], [0.9137155416263443, 0.023574019147635024, -0.40636394703389217], [0.8967775468517001, 0.03943278813570721, -0.4453142783396591], [0.8882094701283203, 0.045657860258455446, -0.4598911611211897], [0.8682172911071009, 0.06018302854486794, -0.4939038876114277], [0.8467970992986515, 0.07574570885173848, -0.5303460945652542], [0.8208947104563968, 0.09101230047740741, -0.5642801771671401], [0.7964203344013714, 0.1052413800826182, -0.5957847793054373], [0.7762299967080366, 0.11635792362645497, -0.6200009400529695], [0.7588956845891617, 0.12539944312937573, -0.639358139873325], [0.728511098713101, 0.14036994607384295, -0.67078409812327], [0.7009825384115005, 0.15326542497739407, -0.6973510954460382], [0.6763100036843597, 0.16408587984002915, -0.7190591318416296], [0.6428712335617238, 0.17772278208283668, -0.7455474852760458], [0.621054724409043, 0.18646821290455565, -0.7623965607444602], [0.591899992648097, 0.1969925790859891, -0.7815964727881111], [0.553979025491655, 0.21033339264759504, -0.8055767018705864], [0.526252306517939, 0.21982024680857043, -0.8223471334506489], [ 0.50092285,  0.22756539, -0.8350391 ]])
comprimento_geodesica5 = 2.114213212694145

geodesica_exata6 = np.array([[-0.3454915028125262, -0.4755282581475767, 0.8090169943749475], [-0.33749083042714173, -0.5247626788553292, 0.7816384650614319], [-0.3309403273934112, -0.5601387702480539, 0.7599304031692063], [-0.32675253479394045, -0.5797416741350839, 0.7464845904606828], [-0.3191645197397518, -0.613689752740579, 0.722347048104869], [-0.3157642972850343, -0.628034927459044, 0.7116553184575792], [-0.30713877021038777, -0.6605549932773092, 0.6850882956381771], [-0.29668816097052986, -0.6969047754769094, 0.6533377092939525], [-0.285200039710214, -0.7318265448892793, 0.6191576424861398], [-0.2744994885946514, -0.7614905851330847, 0.5877316587395607], [-0.26458650762384195, -0.7858968962083245, 0.5590597580542157], [-0.25102336232261, -0.8179626400462349, 0.5200207303192257], [-0.24919828015739884, -0.82179235642757, 0.5148371667944036], [-0.24737319799218746, -0.825622072808905, 0.5096536032695811], [-0.23356011081525083, -0.8510020746910201, 0.47093917813223696], [-0.22339718796873664, -0.8687226438104654, 0.4425918800445374], [-0.21505934728743364, -0.8826134965485755, 0.41942814548165996], [-0.20385891242046128, -0.8989060528807906, 0.3886513669303717], [-0.18979588336781966, -0.9176003128071109, 0.35026154439067264], [-0.17469534229472006, -0.9348665599462013, 0.3094422413873853], [-0.16428247757250092, -0.9459013871098514, 0.2814195458973309], [-0.1510070186646127, -0.9593379178676068, 0.24578380641886605], [-0.13851912990147744, -0.9675167194567973, 0.21290215000163454], [-0.12603124113834224, -0.9756955210459876, 0.1800204935844032], [-0.10964324616907989, -0.9848480134420531, 0.1370963127151732], [-0.09898043957115588, -0.989197098649908, 0.10939821982276404], [-0.08051742056097741, -0.9954935654715137, 0.06161507802635707], [-0.06387948371601015, -0.9979603159117842, 0.01901549975477229], [-0.05504175928329745, -0.9984796847383045, -0.0034990296128145637], [-0.036328798397414, -0.9980904096041148, -0.05095756881157648], [-0.031391180170828675, -0.9976360876237602, -0.06342957372716407], [-0.015540813470614645, -0.9948451088954655, -0.10327506893751498], [0.002384577270515567, -0.9891981045927114, -0.14797952507504308], [0.015372349785060371, -0.9840054222703115, -0.18021197629698338], [0.030435146340521302, -0.9759567143734521, -0.21730338844610084], [0.0475729669368983, -0.9650519809021328, -0.259253761522395], [0.05769814526577384, -0.9574575949856283, -0.2838731687559247], [0.06886083561510734, -0.9484351962818934, -0.3109220564530427], [0.083136062025815, -0.935128759216469, -0.3452593855409261], [0.10259884853881293, -0.9146822582148947, -0.3917441169467518], [0.1199866110108947, -0.89709178278778, -0.4333698874254004], [0.12517417111318488, -0.8899517188516303, -0.44551728974334254], [0.14177436344051347, -0.8671035142559509, -0.48438897716075746], [ 0.15830432, -0.83801025, -0.52218633]])
comprimento_geodesica6 = 1.650597821423333

geodesica_exata7 = np.array([[-0.4755282581475769, -0.3454915028125264, -0.8090169943749473], [-0.4991464452723984, -0.29585613390439053, -0.8144754474629032], [-0.5156474171829635, -0.2586274334328435, -0.816963056813867], [-0.5270258220370158, -0.23173907251875042, -0.8180457833606736], [-0.5420987811603512, -0.19347286002674496, -0.8181039122480493], [-0.5583050110247129, -0.14899896573555485, -0.8164350020739156], [-0.5719499573608183, -0.10969524122309149, -0.814063650497703], [-0.5804723366404108, -0.08073185626808232, -0.8102874161173325], [-0.5889947159200031, -0.05176847131307294, -0.8065111817369617], [-0.5986503659406222, -0.016597404558878703, -0.8010079082950818], [-0.6046114616919579, 0.007195810617403542, -0.7965292325126321], [-0.6117058281843205, 0.03719670759287075, -0.7903235176686733], [-0.6210667361587359, 0.07961296816670838, -0.7806637247016951], [-0.6289996313459214, 0.12306674076100343, -0.768574451271129], [-0.6355045137458769, 0.1675580253757571, -0.7540556973769739], [-0.6397428546637796, 0.19963394639214044, -0.7429910216058381], [-0.6417146540996289, 0.21929450381015347, -0.7353804239577211], [-0.6445249822303016, 0.2524079368469947, -0.721886267722997], [-0.6470405683147706, 0.2927665637034794, -0.7042355919631746], [-0.6484228836582133, 0.32691750876077874, -0.6883119552648621], [-0.648377186214426, 0.3621059658385362, -0.6699588381029608], [-0.6483314887706387, 0.3972944229162936, -0.65160572094106], [-0.6454297657523913, 0.4345579040349672, -0.6283936428519816], [-0.6428227847803476, 0.46457619133399797, -0.6093380842880014], [-0.6384930489748706, 0.5028771844731295, -0.5836965257353354], [-0.6347527972618, 0.5266877899729747, -0.5663680062328644], [-0.6292897907152963, 0.5587811013129211, -0.5424534867417076], [-0.6226935134277662, 0.5846667308536827, -0.5202660063120602], [-0.6160972361402359, 0.610552360394444, -0.4980785258824127], [-0.6080729460654757, 0.6374755019556634, -0.4734615649891767], [-0.5986206432034856, 0.6654361555373411, -0.44641512363235236], [-0.5877403275542654, 0.6944343211394766, -0.41693920181193966], [-0.5725759735433555, 0.7265450228029864, -0.3801748386007616], [-0.5619903999403388, 0.7482979945854786, -0.3548554363054467], [-0.54621179,  0.77382392, -0.32070112]])
comprimento_geodesica7 = 1.3167337671992887

geodesica_exata8 = np.array([[-0.7694208842938134, -0.5590169943749475, -0.30901699437494773], [-0.7836704257961056, -0.5481990238209387, -0.29796593893205087], [-0.8038071466658353, -0.5265362905473681, -0.277297882957495], [-0.8276850832182041, -0.5015428900136752, -0.2533398141475277], [-0.8403393727226547, -0.48654149126034907, -0.23925178384379486], [-0.8619067360489164, -0.4598693610138186, -0.21436573607174048], [-0.8708198098707274, -0.4481986295206145, -0.20356771860341924], [-0.8900765999708817, -0.4198477695612463, -0.17775369186927792], [-0.905592174388396, -0.3948275768620001, -0.15522967797054804], [-0.9225383912624429, -0.36479798718979434, -0.12848767227431981], [-0.9291408918581461, -0.35144852598375265, -0.11676167584391156], [-0.9437765355060852, -0.31974020659870894, -0.08909169118559637], [-0.9546709634713844, -0.2913625544737879, -0.06471171936269299], [-0.968426676349748, -0.2529661084029466, -0.03189576394479232], [-0.977561242775896, -0.21121220290643028, 0.002776149397281935], [-0.9838345242889807, -0.17947709135583367, 0.029012079144359326], [-0.9877972325759573, -0.14606325009239907, 0.0561759878535239], [-0.9908800100933586, -0.10596128214316708, 0.0884858673222737], [-0.9911015026976959, -0.0758781081398549, 0.11235976319602659], [-0.9907722836150766, -0.057492457795300156, 0.12686969651269558], [-0.9886832029933066, -0.025730554079150425, 0.15167157134853534], [-0.987803272223731, -0.019042427393352768, 0.15681754210812082], [-0.9860434106845797, -0.005666174021757786, 0.16710948362729142], [-0.9834036183758531, 0.014398206035634797, 0.18254739590604752], [-0.9766933913018676, 0.04951756917746025, 0.2092052286660613], [-0.9717430257670332, 0.07126067894769039, 0.22557111990690426], [-0.963602156236516, 0.10137064511655602, 0.2480109608694197], [-0.9563412174755741, 0.12479248459962378, 0.2653048310723495], [-0.9458897747189494, 0.15658118048132674, 0.2886726509969516], [-0.9340076895057927, 0.18336047939007016, 0.3078224791240555], [-0.916624527070845, 0.2201853644102863, 0.3339742359349185], [-0.9024318686315804, 0.24864339303186722, 0.35405204302410914], [-0.883618063740101, 0.2804588810791236, 0.37598580803747383], [-0.8624936856225139, 0.3139530988392175, 0.3988475520129252], [-0.834437587826604, 0.3524835057378245, 0.42449323287463736], [-0.8118825672524852, 0.3809683265249586, 0.44313698505259047], [-0.7847064002261511, 0.4128106067377682, 0.4636366951547176], [-0.7676524568738227, 0.43124984141342965, 0.4752785186489116], [-0.7459773670692789, 0.45304653551476615, 0.4887763000672791], [-0.7196811308125204, 0.4782006890417781, 0.5041300394098207], [-0.6795214551991167, 0.5134272208458162, 0.5250516525248838], [-0.677210881973009, 0.5151059505586539, 0.5259796314869708], [-0.64167235281182, 0.5469750229370165, 0.5450452866778599], [-0.637051206359605, 0.5503324823626918, 0.5469012446020338], [-0.6231877670029597, 0.5604048606397178, 0.5524691183745554], [-0.6208771937768521, 0.5620835903525556, 0.5533970973366423], [-0.6139454740985295, 0.5671197794910686, 0.5561810342229031], [-0.57200887,  0.59153968,  0.56823116]])
comprimento_geodesica8 = 1.6383341257793096

geodesica_exata9 = np.array([[0.5590169943749473, -0.7694208842938134, -0.30901699437494773], [0.5436878112823845, -0.7834024180249499, -0.3067941111692682], [0.5132649963269134, -0.8038940618976559, -0.30108946965300726], [0.48284218137144175, -0.8243857057703616, -0.2953848281367461], [0.45960538772662884, -0.8391767953683644, -0.2908743522590286], [0.4399616047371452, -0.8511176078290156, -0.2869609592005829], [0.4078600600688358, -0.869298678475614, -0.2803283387222349], [0.3793515260558561, -0.8846294719848606, -0.2742928010631587], [0.35802901335353454, -0.8942597112194041, -0.269451429042626], [0.336706500651213, -0.9038899504539473, -0.2646100570220933], [0.3065192369253954, -0.9169101707370868, -0.2576465404009301], [0.27992498385490705, -0.9270801138828744, -0.2512801065990387], [0.24805899041625146, -0.937789760939906, -0.24338861101578857], [0.22505774800109246, -0.945109426948342, -0.23761926003316902], [0.20205650558593335, -0.9524290929567778, -0.23184990905054942], [0.18264827382610355, -0.9568984818278621, -0.2266776408872016], [0.14383181030644365, -0.9658372595700301, -0.21633310456050583], [0.11915183817844699, -0.9708463523523585, -0.20963577461579938], [0.0786566449459492, -0.9774745568684193, -0.19836325932701665], [0.06811316420961555, -0.9785539646909074, -0.1953131357642993], [0.03121098163244712, -0.9823318920696165, -0.18463770329478835], [0.010124020159779645, -0.9844907077145932, -0.17853745616935368], [-0.02845689213022662, -0.9859580618671946, -0.16693404473775575], [-0.05122258331573182, -0.9858063042860636, -0.15990581865023404], [-0.07398827450123702, -0.9856545467049327, -0.15287759256271233], [-0.09148222531857542, -0.9849630852125579, -0.1473744282565494], [-0.1211983865850853, -0.9830404583165635, -0.13789316142558203], [-0.15091454785159525, -0.9811178314205691, -0.12841189459461466], [-0.18925990891194716, -0.9751137619836037, -0.1155496080581148], [-0.2084325894421234, -0.9721117272651212, -0.10911846478986484], [-0.24150621013430867, -0.9655679539169115, -0.09778124003472358], [-0.2745798308264938, -0.9590241805687023, -0.08644401527958248], [-0.31101091094435446, -0.9478592607682776, -0.07325083260026732], [-0.34049152098121055, -0.9384652102827167, -0.06251069066439793], [-0.36302166093706195, -0.930842029112019, -0.05422358947197414], [-0.38723053060575086, -0.9209082747152136, -0.045008509317463447], [-0.41143940027443987, -0.9109745203184082, -0.03579342916295268], [-0.43564826994312883, -0.9010407659216029, -0.026578349008442012], [-0.46321459903749373, -0.8864858650725829, -0.015507310929757337], [-0.49078092813185814, -0.8719309642235624, -0.004436272851072759], [-0.5200259869390601, -0.8550654901484342, 0.007562744189698767], [-0.5526285051719376, -0.8335788696210911, 0.021417719154644174], [-0.5749230938981352, -0.8184842648608267, 0.030963695451970072], [-0.6005751420500083, -0.7987685136483471, 0.042365629673469865], [-0.6142405309823634, -0.7877553514290534, 0.04853058626526317], [-0.6432500385599117, -0.7634184537643587, 0.06178847841093678], [-0.6602728869179422, -0.7477841450928497, 0.069809392926904], [-0.6772957352759732, -0.7321498364213415, 0.07783030744287123], [-0.6993547727725167, -0.7095838080715099, 0.0886351588450992], [-0.7247712696947359, -0.6823966332694639, 0.10129596817150108], [-0.7281287291204114, -0.6777754868172488, 0.10315192609567496], [-0.7552239557554679, -0.6482777387890947, 0.1167407143841637], [-0.780404901448033, -0.6136191403974814, 0.13066039881546765], [-0.8039071174277607, -0.5812711152319756, 0.14365210428468472], [-0.8221375930393214, -0.5483833861552261, 0.15511874797254305], [-0.8420467983637194, -0.5131850838523685, 0.16751337062248822], [-0.8569198145496049, -0.48491850122783386, 0.17712405638617273], [-0.8667566415969769, -0.4635836382816214, 0.18395080526359642], [-0.8765934686443492, -0.44224877533540957, 0.1907775541410201], [-0.8912309336005805, -0.4065107691213081, 0.20164711500960658], [-0.8977103012222776, -0.3897970526273113, 0.20661790596285645], [-0.9089903067528334, -0.35868019286542474, 0.21563150890726918], [-0.9169128528577141, -0.33218447955575336, 0.22278915392750798], [-0.9279573071586164, -0.2935961962043001, 0.2330616319768228], [-0.9356443020338433, -0.25962905930506214, 0.24147815210196377], [-0.9416525671962325, -0.22797249563193162, 0.2489666932650178], [-0.9476608323586213, -0.19631593195880065, 0.2564552344280717], [-0.9503116380953349, -0.1692805147378854, 0.26208781766695183], [-0.952962443832049, -0.14224509751697, 0.26772040090583205], [-0.9556132495687628, -0.11520968029605455, 0.2733529841447122], [-0.956585325592639, -0.09048483630124671, 0.2780575884215054], [-0.957321850386861, -0.058288568716871825, 0.2840210678032007], [-0.9578228239514295, -0.018620877542930292, 0.29124342228979816], [-0.95487527,  0.01630032,  0.29655948]])
comprimento_geodesica9 = 2.2641405045732204

geodesica_exata10 = np.array([[1.0797439917089951e-16, 0.5877852522924732, -0.8090169943749473], [0.014188498772236782, 0.6072611331095069, -0.7967228966151824], [0.03383411245687219, 0.6312406096784293, -0.7755888712402046], [0.05129688017654823, 0.6523595754199432, -0.7565331286331585], [0.06439395596630523, 0.66775751950664, -0.7416339515619771], [0.07749103175606223, 0.6831554635933366, -0.7267347744907957], [0.08622241561590016, 0.6928323860252158, -0.7159921629554785], [0.10368518333557605, 0.7121862308889744, -0.6945069398848442], [0.11896510509029257, 0.7286795649253246, -0.6750999995821424], [0.12551364298517098, 0.7354959765297951, -0.6664356708147573], [0.142976410704847, 0.7530847005157988, -0.6425209672805345], [0.1538906405296445, 0.7638570128973315, -0.6272705925136968], [0.16917056228436084, 0.7785852260559264, -0.6054341717474065], [0.17790194614419896, 0.7864970276100506, -0.5922620797485008], [0.1866333300040369, 0.7944088291641747, -0.5790899877495951], [0.20191325175875324, 0.8073719214450144, -0.5548240865197165], [0.21282748158355072, 0.8163791129487918, -0.53714423128929], [0.23029024930322672, 0.8304375951792851, -0.5083705668278905], [0.2455701710579431, 0.8416355665823696, -0.48167518513442314], [0.2543015549177812, 0.8477822472587387, -0.4660736126719291], [0.2652157847425787, 0.8550243178847614, -0.44596427697791474], [0.27831286053233567, 0.8633617784604373, -0.4213471780523796], [0.29359278228705205, 0.8727946289857663, -0.3922223158953237], [0.30668985807680904, 0.8793669686836872, -0.36517573650620017], [0.31978693386656604, 0.8859393083816078, -0.33812915711707625], [0.3350668556212823, 0.8918419171514267, -0.30414533403284383], [0.3481639314110393, 0.8966491359715922, -0.2746692741801317], [0.3568953152708774, 0.8992655748924511, -0.2542087407904607], [ 0.36344385,  0.90067516, -0.23810255]])
comprimento_geodesica10 = 0.7644827942154905


## PLOT DAS GEODÉSICAS E DA ESFERA

# Plot das geodésicas exatas
ax = plt.axes(projection="3d")

plt.plot(geodesica_exata1[:,0], geodesica_exata1[:,1], geodesica_exata1[:,2], color="blue")
plt.plot(geodesica_exata2[:,0], geodesica_exata2[:,1], geodesica_exata2[:,2], color="blue")
plt.plot(geodesica_exata3[:,0], geodesica_exata3[:,1], geodesica_exata3[:,2], color="blue")
plt.plot(geodesica_exata4[:,0], geodesica_exata4[:,1], geodesica_exata4[:,2], color="blue")
plt.plot(geodesica_exata5[:,0], geodesica_exata5[:,1], geodesica_exata5[:,2], color="blue")
plt.plot(geodesica_exata6[:,0], geodesica_exata6[:,1], geodesica_exata6[:,2], color="blue")
plt.plot(geodesica_exata7[:,0], geodesica_exata7[:,1], geodesica_exata7[:,2], color="blue")
plt.plot(geodesica_exata8[:,0], geodesica_exata8[:,1], geodesica_exata8[:,2], color="blue")
plt.plot(geodesica_exata9[:,0], geodesica_exata9[:,1], geodesica_exata9[:,2], color="blue")
plt.plot(geodesica_exata10[:,0], geodesica_exata10[:,1], geodesica_exata10[:,2], color="blue")

#plot da esfera
precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,np.pi,precisao)
x=np.outer(np.cos(u),np.cos(v))
y=np.outer(np.cos(u),np.sin(v))
z=np.outer(np.sin(u),np.ones(precisao))

ax.plot_surface(x,y,z, alpha=0.5, color='white')
ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")





## COMPARAÇÃO 

# mudar seed para variar as amostras de pontos
rng = np.random.default_rng(3)

# cada elemento das listas de erros é a média dos erros calculados para um certo número de amostras 
erros_abs_medios = []
erros_relativos_medios = []


# lista com as quantidades de amostras usadas. Será usada para plotar o gráfico da comparação
amostras = []

# listas com os pontos iniciais e finais de cada geodésica, para serem usados como pontos iniciais e finais dos caminhos mínimos
inicios = [geodesica_exata1[0], geodesica_exata2[0], geodesica_exata3[0], geodesica_exata4[0], geodesica_exata5[0], geodesica_exata6[0], geodesica_exata7[0], geodesica_exata8[0], geodesica_exata9[0], geodesica_exata10[0]]
fins = [geodesica_exata1[len(geodesica_exata1) - 1], geodesica_exata2[len(geodesica_exata2) - 1], geodesica_exata3[len(geodesica_exata3) - 1], geodesica_exata4[len(geodesica_exata4) - 1], geodesica_exata5[len(geodesica_exata5) - 1], geodesica_exata6[len(geodesica_exata6) - 1], geodesica_exata7[len(geodesica_exata7) - 1], geodesica_exata8[len(geodesica_exata8) - 1], geodesica_exata9[len(geodesica_exata9) - 1], geodesica_exata10[len(geodesica_exata10) - 1]]

comprimentos = [comprimento_geodesica1, comprimento_geodesica2, comprimento_geodesica3, comprimento_geodesica4, comprimento_geodesica5, comprimento_geodesica6, comprimento_geodesica7, comprimento_geodesica8, comprimento_geodesica9, comprimento_geodesica10]




## INÍCIO DA COMPARAÇÃO
amostra = 30
for l in range(29):
  amostra += 50
  X = rng.uniform(0,2*np.pi,size=(amostra,2))


  ## Imagem de X pela parametrização, com os pontos iniciais e finais adicionados
  Im_X = []
  for i in range(10):
    Im_X.append(inicios[i])
    Im_X.append(fins[i])

  for ponto in X:
    Im_X.append([np.cos(ponto[0])*np.cos(ponto[1]), np.cos(ponto[0])*np.sin(ponto[1]), np.sin(ponto[0])])
  Im_X = np.array(Im_X)


  ## Aplicação do KNN em Im_X
  numero_vizinhos = 8
  knn = NearestNeighbors(n_neighbors=numero_vizinhos)
  knn.fit(Im_X)
  distancias, vizinhos = knn.kneighbors(Im_X,return_distance=True)

  
  ## Construção do grafo
  grafo = np.zeros((amostra+20,amostra+20))

  for i in range(len(Im_X)):
    for j in range(len(Im_X)):

      if j in vizinhos[i]:
        for k in range(numero_vizinhos):
          if vizinhos[i][k] == j:
            grafo[i,j] = distancias[i,k]


  G = nx.from_numpy_array(grafo)
  
  ## Variação entre as geodésicas fixadas
  # Índices dos pontos inicial e final do caminho mínimo (ver lista Im_X). Serão atualizados para variar a geodésica usada
  inicio = 0
  fim = 1
  erro_abs_medio = 0
  erro_relativo_medio = 0
  for j in range(10):

    #Dijkstra da NetworkX
    caminho = nx.dijkstra_path(G,inicio,fim)
    
    inicio += 2
    fim += 2

    
    # Plot dos caminho mínimos
    # usar l e j para controlar quais caminhos mínimos serão plotados
    if l==19:
      for i in range(len(caminho)-1):
        x = [Im_X[caminho[i],0], Im_X[caminho[i+1],0]]
        y = [Im_X[caminho[i],1], Im_X[caminho[i+1],1]]
        z = [Im_X[caminho[i],2], Im_X[caminho[i+1],2]]

        ax.plot(x,y,z,color='red')
      #ax.set_aspect('equal')  
    

    
    # Comparação dos comprimentos
    comprimento_geodesica = comprimentos[j]
    comprimento_caminho = 0
    for i in range(1,len(caminho)):
      comprimento_caminho += np.linalg.norm(Im_X[caminho[i]] - Im_X[caminho[i-1]]) 
 

    erro_abs_medio += comprimento_caminho - comprimento_geodesica
    erro_relativo_medio += (comprimento_caminho - comprimento_geodesica)/comprimento_geodesica



  erros_abs_medios.append(erro_abs_medio/10)
  erros_relativos_medios.append(erro_relativo_medio/10)

  # Somamos 20 ao número de amostras por causa dos pontos de inicio e fim adicionados a Im_X.
  amostras.append(amostra + 20)


## Plot dos erros
fig, ax = plt.subplots()
plot1 = ax.plot(amostras, erros_relativos_medios, color='blue', label='Erro relativo médio')
plot2 = ax.plot(amostras, erros_abs_medios, color='red', label='Erro absoluto médio')

ax.legend()
#ax.set_title('Progressão do erro dos caminhos mínimos com o aumento da amostra')
ax.set_xlabel('tamanho da amostra')
ax.set_ylabel('Erro')

plt.show()