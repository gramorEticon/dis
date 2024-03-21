import matplotlib.pyplot as plt

ml100k = [[0.4540199973080089, 0.45435034799320545, 0.45725575680422287, 0.45865926218858916, 0.4610286129629372,
      0.45999312359239297, 0.46159160191865545, 0.46214535946361734, 0.46291216881829483, 0.46287726568657306],
     [0.5914540019690391, 0.5943054055872372, 0.5942488868209067, 0.5954882469894065, 0.5943421474498471,
      0.5934346590376788, 0.5930060741069749, 0.5931246076362637, 0.5926073718018995, 0.5911509333843143],
     [0.6208686161012178, 0.6231877366824573, 0.6233127697059435, 0.6230952514606243, 0.6238058745116908,
      0.6214420958642982, 0.6198349370026781, 0.6195044376577926, 0.6173458392895823, 0.6168479858590454],
     [0.6265774864780007, 0.6275977042760839, 0.6271774718870448, 0.6260255133729457, 0.6234423169165334,
      0.6241700270962159, 0.62323636871155, 0.6218821990800063, 0.6225406542946457, 0.623148921579721],
     [0.6261489882310175, 0.6268173099109222, 0.6245188863643466, 0.6239690428593828, 0.6230690566980834,
      0.6223739839455014, 0.6203090367674581, 0.6222465249262362, 0.6222479689223644, 0.621364471950892],
     [0.6334758599228472, 0.6337479579003452, 0.6335304777677209, 0.6325973050706021, 0.630999468200949,
      0.6298050016687695, 0.6281796433679514, 0.62638947106408, 0.6261962935053771, 0.6236086598815229],
     [0.6312979736325293, 0.6300910323724608, 0.6285540082755569, 0.6285667898777033, 0.6289295825260859,
      0.6269389754665079, 0.625282204965568, 0.6248507682117249, 0.62443579738994, 0.6251122671247212],
     [0.6334261875108889, 0.6334356168663253, 0.6326853050939912, 0.6337312363077622, 0.6328026903793562,
      0.6347150692465618, 0.6334581480935397, 0.6343371725666277, 0.633471919839142, 0.63227369056207],
     [0.6363308462762614, 0.6372894510344925, 0.6339196266533799, 0.6343743793481943, 0.6329454889122942,
      0.6333828037669681, 0.6357244593352641, 0.6322312377767111, 0.6299756865094447, 0.6305802302325165],
     [0.6327786030873069, 0.6287118328123997, 0.6302221851270948, 0.6281818828615882, 0.6290665542408379,
      0.6296705231174826, 0.6296490265172755, 0.6288903271022952, 0.6293410729624266, 0.6289818272133244],
     [0.6297568302770846, 0.6265312956326604, 0.626855178172013, 0.6305140912326275, 0.6290057892114714,
      0.6274184700786087, 0.6267583397922236, 0.6261954480800974, 0.6257348334083889, 0.6257635746987714],
     [0.6280134856760129, 0.6273083927526604, 0.6325660103703328, 0.6310344320559765, 0.6328958824366804,
      0.633171498263725, 0.6301960045147201, 0.6312039554398176, 0.6298924131541155, 0.6288618623336313],
     [0.6452908930125077, 0.6453816129144081, 0.6469854433663973, 0.6476080639368315, 0.6475789442271186,
      0.646216459834212, 0.6441202434775111, 0.6453276005074993, 0.6441217155513655, 0.6418374020574823],
     [0.6472937512142043, 0.6459095869747975, 0.640765767947594, 0.6427100642730735, 0.6415444792592098,
      0.6414685825759073, 0.6405589510933247, 0.6394250666099529, 0.6388674836533993, 0.6378101371071813],
     [0.632114283670273, 0.6392270684983266, 0.6399579627716306, 0.6382567582259406, 0.6383664120435408,
      0.6379755151673419, 0.6322364495464795, 0.6318878959078489, 0.6316531317247849, 0.6338200690614161],
     [0.6295973594694249, 0.6293407633105129, 0.6283796233124962, 0.6265699564611305, 0.6268025159847015,
      0.6265612206331714, 0.6291934247178121, 0.6259776773632391, 0.6264542733081265, 0.6258575170009683],
     [0.6279716937958233, 0.6275057625076506, 0.6328904124880592, 0.631280993025369, 0.6326773786848251,
      0.6314453868366937, 0.6340264964801747, 0.6333174413068094, 0.631847621429299, 0.6346366834254431],
     [0.6369728792163287, 0.6347830374733162, 0.6351900466588397, 0.6327529493704365, 0.6349147225410655,
      0.6350735313417951, 0.6317124106112829, 0.6315319849944393, 0.6256486636849761, 0.6333205102834735],
     [0.6365624749260534, 0.6342221959965245, 0.6357484446468656, 0.6331261447674918, 0.6343929698849137,
      0.6348567768684652, 0.6342012608605527, 0.633316367361675, 0.6346429556864452, 0.6340682949575953],
     [0.6328778992032713, 0.6348734740973275, 0.6356821179589094, 0.6360049278637921, 0.6352395385056273,
      0.634119699959772, 0.6343668628878836, 0.6343291784222652, 0.6326934875840272, 0.6317581664390433],
     [0.6198488901216902, 0.6261094884449495, 0.6251749189296804, 0.6249231061657375, 0.6258545799441693,
      0.6261759573050909, 0.6241935732177044, 0.6255151335534829, 0.6234803983127587, 0.6240936367162293],
     [0.6256538959621459, 0.6280542751306842, 0.6290739682996679, 0.6277727951461226, 0.6283956745030258,
      0.6274958982931559, 0.6256784351371357, 0.6267759134773236, 0.6291077658096219, 0.627075817046781],
     [0.6305078496503682, 0.6282681489276507, 0.628815057009504, 0.6279806912920664, 0.6293271392694627,
      0.6294255150033201, 0.6317598596168885, 0.6308938863311834, 0.6276350663754874, 0.6232369799878587],
     [0.6207799922842315, 0.6231749552032523, 0.6295380966701496, 0.6271865105449983, 0.6277407512565069,
      0.6288556778359861, 0.627061074346888, 0.6241086851305659, 0.6244868019914882, 0.624768113286078],
     [0.6066351527554693, 0.6186976490680389, 0.6191342965910842, 0.6249208216139629, 0.6246653052625997,
      0.6281854960606384, 0.6285883465648826, 0.6248685828898352, 0.6294463601410565, 0.6291381383565284],
     [0.6188734926743595, 0.6311688211568584, 0.6287496516597397, 0.6247573091931516, 0.6267865156658701,
      0.6294284338361646, 0.6252548325466465, 0.6274286689977332, 0.6286388335403338, 0.6249511365028994],
     [0.6161098075402753, 0.6235489587946823, 0.6226680915147822, 0.6261395132355017, 0.6264438691678351,
      0.6188481501843998, 0.6233294855186429, 0.6229212498123424, 0.6143293110323996, 0.622223691386602],
     [0.6192380718949784, 0.6084802533617625, 0.6159524569854743, 0.6176759656290173, 0.6165075897875528,
      0.6150423850264254, 0.6165484720458353, 0.6157848349766782, 0.6138032530082408, 0.6133078868110284],
     [0.6141027397548078, 0.6121189376852648, 0.6114191170905734, 0.6112609044459875, 0.6121791662453999,
      0.6112625700405074, 0.6101391562240603, 0.611994784971028, 0.6099977679792591, 0.610251899528598],
     [0.6092860746449422, 0.6072390302975891, 0.6145173437814552, 0.6150804860459645, 0.6131521147922592,
      0.6158139370053614, 0.6166021842098743, 0.6168311629795096, 0.6169339856204735, 0.6174876696386338],
     [0.6112336845299982, 0.6094134745994808, 0.6162199168575898, 0.6169764443343595, 0.6176032158786874,
      0.6185431381601423, 0.6198202939613435, 0.6174885608318791, 0.618500544551759, 0.6177526499923285],
     [0.6140071047464983, 0.6174065004221164, 0.6147247003289568, 0.6114058216274791, 0.6110062756448139,
      0.6126958182044022, 0.61309922198595, 0.613304460403676, 0.6141984078059681, 0.6145744905579337],
     [0.6147367208653635, 0.6141291497416925, 0.6151111102366664, 0.6141881335060112, 0.6142774329024614,
      0.6135169255101005, 0.612840178701386, 0.6147129674924511, 0.61485966326161, 0.6151911541060269],
     [0.601744443369573, 0.6030494080291118, 0.6006078213065197, 0.6105380142862252, 0.6098556983365202,
      0.608166336557297, 0.6003499464411731, 0.6060222209412445, 0.6091423936079136, 0.6097386985571576],
     [0.6078973699914031, 0.6080523194055563, 0.6082218285482305, 0.6071683938879753, 0.609122343129514,
      0.608404234660267, 0.6092133650615661, 0.6092140810002818, 0.6098621771113483, 0.6119024739431009],
     [0.6038188885565972, 0.6060820476862583, 0.6054874926909415, 0.6043254158589558, 0.6042285007866249,
      0.6036016017826389, 0.6026449527473557, 0.600631070009997, 0.6029198949691748, 0.6041938376121773],
     [0.5995899826987965, 0.6014980181319213, 0.5976392793873678, 0.5980099748803333, 0.5951559835040034,
      0.5958010577760164, 0.5946709240481373, 0.5958074217812085, 0.5962669288312396, 0.5962390887840301],
     [0.5961566056402109, 0.593691951131581, 0.5919992546277949, 0.5935099842482225, 0.5930753964836198,
      0.593772865899214, 0.596802196058354, 0.5952795810942172, 0.5966828410487293, 0.594292055812641],
     [0.5933233953043925, 0.5931981824369582, 0.593625512482131, 0.5926466070781707, 0.5940388980790668,
      0.5942078029530409, 0.5950721404943689, 0.5945336222398125, 0.5935104009258279, 0.5954058524799538],
     [0.5753862766247426, 0.5883303699182236, 0.5924397999890658, 0.5947882164014058, 0.5936583171700902,
      0.593451671698185, 0.5935779989519334, 0.5935153716378478, 0.5938865465854735, 0.5954499430989768],
     [0.5968540648733932, 0.5964231442941978, 0.5948527252800597, 0.5982392213089226, 0.596839914703181,
      0.5962557311775529, 0.599528662216921, 0.5973459050506751, 0.598422137624665, 0.5965478909529582],
     [0.5874337999539936, 0.5873322626743169, 0.5881208647607794, 0.5877967508636046, 0.5898232082772092,
      0.58820269173998, 0.5907830994903771, 0.5907524125381147, 0.5901847303932122, 0.588883017302684],
     [0.5910016291529538, 0.5938403185138229, 0.5935591948454554, 0.5904386968880463, 0.5896735763762435,
      0.5925253426749995, 0.5895606127522961, 0.5921017126599512, 0.5889378073583449, 0.5906718677660949],
     [0.5903080251526823, 0.5889130835241161, 0.5891878637087296, 0.5853413095835682, 0.5868419435117491,
      0.5867116776833586, 0.5854771381671164, 0.5870354485320143, 0.5871697011821742, 0.5884004296689509],
     [0.5842711849954061, 0.5842140233756136, 0.5826469947164499, 0.5845055955687999, 0.5841789473221727,
      0.5831269814859368, 0.5831112992599599, 0.5798720881481292, 0.582547154087526, 0.5813195962291791],
     [0.5836869703408676, 0.5831370187242033, 0.58250684858373, 0.5824443533794302, 0.5825324847923031,
      0.5825089534583944, 0.5826992535198218, 0.5827301544450426, 0.5831109616200555, 0.582893012398138],
     [0.5780005198701414, 0.5777554756587614, 0.5760349983764284, 0.5757184674328758, 0.5767769126090804,
      0.5778861104997688, 0.5775833421800352, 0.5782031662043564, 0.5779708947034452, 0.578066355483553],
     [0.5808140126048958, 0.5798770571128629, 0.5796671569122962, 0.5620495267318993, 0.5779547337533055,
      0.5791719643215886, 0.5781742872222296, 0.5770290932364135, 0.5778725471328102, 0.5779886494660019],
     [0.5702146973802205, 0.5762072118714692, 0.5768399801677244, 0.5715031234067678, 0.5763948793303223,
      0.5764048261688608, 0.5714514985408606, 0.5771028656665376, 0.5770870959176619, 0.5743125781845834],
     [0.5730842547303922, 0.571480641348073, 0.5675304012555017, 0.5726099530062922, 0.5730842009073814,
      0.5695350865821514, 0.5732708953720558, 0.5743855389621434, 0.5689366023199947, 0.5753049125761858]]
#
# merged = [element for each_list in ml100k for element in each_list]
# print(max(merged))


plt.imshow(ml100k, cmap='bone', interpolation='nearest')
plt.show()