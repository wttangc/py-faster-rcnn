from os import listdir, mkdir
from os.path import isdir, isfile, basename, splitext
import re
from PIL import Image, ImageDraw, ImageFont
from string import Template
import scipy.io as sio
import numpy as np
from copy import copy

def randchoose(ar):
    if len(ar) == 0:
        print "Empty array when calling randchoose()"
        exit(0)
        
    idx = np.random.choice( len(ar), 1 )[0]
    return idx, ar[idx]

def firstLastAbove(ar, thres):
    firstIdx = len(ar)
    lastIdx = -1
    
    for i in xrange(len(ar)):
        if ar[i] >= thres:
            firstIdx = i
            break
    for i in xrange(len(ar)):
        if ar[-i-1] >= thres:
            lastIdx = len(ar) - i - 1
            break
            
    return firstIdx, lastIdx
                
def getFeasBoxes(sols, curSol, w, h, constraintBoxes):
    curConstraintBox = constraintBoxes[0]
    updatedConstraintBoxes = constraintBoxes[1:]

    # x1b <= x2a
    updatedSol = copy(curSol)
    updatedSol[2] = min( updatedSol[2], curConstraintBox[0] )
    if updatedSol[2] - updatedSol[0] >= w:
        if len(updatedConstraintBoxes) > 0:
            getFeasBoxes(sols, updatedSol, w, h, updatedConstraintBoxes)
        else:
            sols.append(updatedSol)
    
    # x1a >= x2b
    updatedSol = copy(curSol)
    updatedSol[0] = max( updatedSol[0], curConstraintBox[2] )
    if updatedSol[2] - updatedSol[0] >= w:
        if len(updatedConstraintBoxes) > 0:
            getFeasBoxes(sols, updatedSol, w, h, updatedConstraintBoxes)
        else:
            sols.append(updatedSol)

    # y1b <= y2a
    updatedSol = copy(curSol)
    updatedSol[3] = min( updatedSol[3], curConstraintBox[1] )
    if updatedSol[3] - updatedSol[1] >= h:
        if len(updatedConstraintBoxes) > 0:
            getFeasBoxes(sols, updatedSol, w, h, updatedConstraintBoxes)
        else:
            sols.append(updatedSol)

    # y1a >= y2b
    updatedSol = copy(curSol)
    updatedSol[1] = max( updatedSol[1], curConstraintBox[3] )
    if updatedSol[3] - updatedSol[1] >= h:
        if len(updatedConstraintBoxes) > 0:
            getFeasBoxes(sols, updatedSol, w, h, updatedConstraintBoxes)
        else:
            sols.append(updatedSol)
    
    return

def getFeasBox(canvasW, canvasH, margin, w, h, constraintBoxes):
    sols = []
    initSol = [ margin, margin, canvasW - margin, canvasH - margin ]
    if len(constraintBoxes) == 0:
        sol = initSol
    
    else:    
        getFeasBoxes(sols, initSol, w, h, constraintBoxes)
        if len(sols) == 0:
            return None
   
        solIdx = np.random.choice( len(sols), 1 )[0]
        sol = sols[solIdx]
        #print "Pick sol %d from %d solutions" %(solIdx + 1, len(sols))
    
    if sol[2] - sol[0] - w > 0:    
        deltaX = np.random.choice( sol[2] - sol[0] - w, 1 )[0]
    else:
        deltaX = 0
    
    xmin = sol[0] + deltaX
    xmax = xmin + w
    
    if sol[3] - sol[1] - h > 0:
        deltaY = np.random.choice( sol[3] - sol[1] - h, 1 )[0]
    else:
        deltaY = 0
        
    ymin = sol[1] + deltaY
    ymax = ymin + h
    return [ xmin, ymin, xmax, ymax ]

overlapTol = 24
back_ims = [ ]
back_count = 0
back_filenames = [ ]
categories = [ ]
cat2cid = { }
cid_ims = [ ]
#cid_maskIms = [ ]
#cid_breadBoxes = [ ]
cid_count = [ ]
cid_filenames = [ ]

nextCid = 0
    
def loadImage(breadPath, category):
    global nextCid
    breadIm = Image.open(breadPath)
    breadFilename = basename(breadPath)
    breadBasename = splitext(breadFilename)[0]

    #print "0,0,%d,%d => %d,%d,%d,%d" %( breadIm.size[0], breadIm.size[1], firstDenseRow, 
    #                                    firstDenseCol, lastDenseRow, lastDenseCol )
    
    if category not in cat2cid:
        cat2cid[category] = nextCid
        cid_ims.append([ np.array(breadIm) ])
        cid_count.append(1)
        categories.append(category)
        cid_filenames.append( [breadBasename] )
        nextCid += 1
    else:
        cid = cat2cid[category]
        cid_ims[cid].append( np.array(breadIm ))
        cid_count[cid] += 1
        cid_filenames[cid].append(breadBasename)
    
    breadIm.close()

def trimIm(breadIm):
    densePixRatio = 0.25
    greyBread = breadIm.convert("L")
    breadMask = np.array(greyBread) != 0
    rowPixCount = np.sum(breadMask, axis=1)
    colPixCount = np.sum(breadMask, axis=0)
    firstFilledRow, lastFilledRow = firstLastAbove(rowPixCount, 1)
    firstFilledCol, lastFilledCol = firstLastAbove(colPixCount, 1)
    breadIm2 = breadIm.crop((firstFilledCol, firstFilledRow, 
                             lastFilledCol+1, lastFilledRow+1))
    w, h = breadIm2.size
    greyBread.close()
    
    greyBread2 = breadIm2.convert("L")
    breadMask2 = np.array(greyBread2) != 0
    breadMaskIm2 = Image.fromarray( np.array(breadMask2, np.uint8) * 255, "L" )
    greyBread2.close()
    
    rowPixThres2 = w * densePixRatio
    colPixThres2 = h * densePixRatio
    rowPixCount2 = np.sum(breadMask2, axis=1)
    colPixCount2 = np.sum(breadMask2, axis=0)
    firstDenseRow, lastDenseRow = firstLastAbove(rowPixCount2, rowPixThres2)
    firstDenseCol, lastDenseCol = firstLastAbove(colPixCount2, colPixThres2)

    breadBox = [ firstDenseCol + overlapTol, firstDenseRow + overlapTol, 
                 lastDenseCol - overlapTol,  lastDenseRow - overlapTol ]
    return breadIm2, breadMaskIm2, w, h, breadBox
    
def loadBGImage(bgImagePath, batchname):
    global back_count, back_ims, back_filenames
    bgIm = Image.open(bgImagePath)
    bgImageBasename = splitext(bgImagePath)[0]
    back_ims.append( np.array(bgIm) )
    back_filenames.append( batchname + "_" + bgImageBasename )
    back_count += 1
                                                    
datapath = "d:/bread-data/"
imgSaveDir = datapath + "JPEGImages"
gtImgSaveDir = datapath + "GroundTruth"
annotSaveDir = datapath + "Annotations"
imgSetsSaveDir = datapath + "ImageSets"
proposalDir = datapath + "Proposals"
annotBasicTmplFile = open(datapath + 'annot-basic.xml')
annotObjTmplFile = open(datapath + 'annot-obj.xml')
annotBasicTmpl = Template( annotBasicTmplFile.read() )
annotObjTmpl = Template( annotObjTmplFile.read() )

if not isdir(imgSaveDir):
    mkdir(imgSaveDir)
if not isdir(annotSaveDir):
    mkdir(annotSaveDir)
if not isdir(imgSetsSaveDir):
    mkdir(imgSetsSaveDir)
if not isdir(proposalDir):
    mkdir(proposalDir)

count = 0
imglist = []
all_proposals = []
MaxImgPerCat = 10000
filterCats = True
whiteCats = { 'FireFlosss': 1, 'Flosss': 1, 'SeaweedFlosss': 1, 'UNDEFBG': 1 }
        
for batchname in listdir(datapath):
    if isdir(batchname) and re.match("^(btg)?[0-9_]+(On|Off)?$", batchname):
        batchdir = datapath + "/" + batchname
        match = re.match("^(btg)?[0-9_]+(On|Off)?$", batchname)
        onoff = match.group(2)
        # currently ignore dark images
        if onoff and onoff.lower() == "off":
            continue
            #onoff = "_" + onoff
        
        print batchname
            
        for category in listdir(batchdir):
            if category in cat2cid and cid_count[ cat2cid[category] ] == MaxImgPerCat:
                continue
            # if not "white" cats (whitelisted categories), discard
            if filterCats and category not in whiteCats:
                continue
                
            catdir = batchdir + "/" + category
            for imgFilename in listdir(catdir):
                # skip depth images
                if 'depth' in imgFilename:
                    continue
                imagePath = catdir + "/" + imgFilename
                if category == 'UNDEFBG':
                    loadBGImage(imagePath, batchname)
                else:
                    loadImage(imagePath, category)

                count += 1     
                if count %100 == 0:
                    print "\r%d" %count

print "\nTotal %d images loaded" %count
print "%d categories:" %len(categories)
for cid,cat in enumerate(categories):
    print "%s\t:%d" %(cat, cid_count[cid])
print "Background: %d" %back_count

# categories except tray
K = len(categories)
ImgNum = 10000
imgStartId = 15000
appendMode = True
ObjNum = 7
canvW = 640
canvH = 480
canvSize = (canvW, canvH)
trayMargin = 20
img_gtBoxes = []
angles = range(0,360,30)
font = ImageFont.truetype("times.ttf", 14)
lumaScaleRange = (0.8, 1.2)

for i in xrange(imgStartId, imgStartId + ImgNum):
    if (i+1)%100 == 0:
        print "\r%d\r" %(i+1),
        
    #print "Multi-bread image %d:" %(i+1)
    bgImgIdx, bgIm = randchoose(back_ims)
    bgIm = Image.fromarray(bgIm)
    back_filename = back_filenames[bgImgIdx]
    #multIm = Image.new( "RGB", canvSize, (230,230,230) )
    multIm = bgIm.resize(canvSize, Image.LANCZOS)
    constraintBoxes = []
    selCats = []
    selFilenames = []
    
    for k in xrange(ObjNum):
        cid = np.random.choice(K, 1)[0]
            
        imgIdx, breadIm = randchoose(cid_ims[cid])
        breadIm = Image.fromarray(breadIm)
        angleIdx, angle = randchoose(angles)
        breadImRot = breadIm.rotate(angle, expand=True)
        breadIm2, breadMaskIm2, breadW, breadH, breadBox = trimIm(breadImRot)
        breadIm.close()
        breadImRot.close()
        box = getFeasBox( canvW, canvH, trayMargin, 
                            breadW, breadH, constraintBoxes )
        # no room for this bread, but there may be room for the next random bread
        if not box:
            continue
            #print "No room for image %d (%dx%d)" %(k+1, breadW, breadH)
            #break
        
        #print "Image %d put at %s" %(k+1, str(box))
        multIm.paste( breadIm2, (box[0], box[1]), breadMaskIm2 )
        selCats.append( categories[cid] )
        selFilenames.append( cid_filenames[cid][imgIdx] )
        constraintBoxes.append( [ breadBox[0] + box[0], breadBox[1] + box[1], 
                                  breadBox[2] + box[0], breadBox[3] + box[1] ] )
    
    multImgFileBasename = "%05d" %(i+1)
    imglist.append(multImgFileBasename)
    gtBoxes = np.array( constraintBoxes, dtype=np.uint16 )
    img_gtBoxes.append(gtBoxes)

    # change lighting in the Y channel of the YCbCr color space
    ycbcrIm = multIm.convert("YCbCr")
    ycbcrAr = np.array(ycbcrIm)
    
    lumaScale = ( np.random.random_sample() - 0.5 ) * \
                    ( lumaScaleRange[1] - lumaScaleRange[0] ) + \
                    ( lumaScaleRange[0] + lumaScaleRange[1] ) / 2

    lumas = np.minimum( ycbcrAr[:,:,0] * lumaScale, 255)
    ycbcrAr[:,:,0] = lumas.astype("uint8")
    ycbcrIm2 = Image.fromarray(ycbcrAr, "YCbCr")
    multIm2 = ycbcrIm2.convert("RGB")
    ycbcrIm.close()
    ycbcrIm2.close()
    multIm.close()
    
    multIm3 = multIm2.copy()
    imDrawer = ImageDraw.Draw(multIm3)

    basicVars = { 'filename': multImgFileBasename, 'back_filename': back_filename, 'lumascale': lumaScale }
    annotation = annotBasicTmpl.substitute(basicVars)
    
    for k, cat in enumerate(selCats):
        xmin, ymin, xmax, ymax = constraintBoxes[k]
        xmin = max( trayMargin/2, xmin - overlapTol/2 )
        ymin = max( trayMargin/2, ymin - overlapTol/2 )
        xmax = min( canvW - trayMargin/2, xmax + overlapTol/2 )
        ymax = min( canvH - trayMargin/2, ymax + overlapTol/2 )
        
        imDrawer.rectangle((xmin, ymin, xmax, ymax))
        imDrawer.text((xmin, ymin), cat, font=font, fill=(255,0,0))
        selFilename = selFilenames[k]
        objVars = { 'category': cat, 'srcfilename': selFilename,
                    'xmin': xmin, 'xmax': xmax, 
                    'ymin': ymin, 'ymax': ymax }
        annotObj = annotObjTmpl.substitute(objVars)
        annotation += annotObj
    annotation += "</annotation>\n"
        
    annotFile = open(annotSaveDir + "/" + multImgFileBasename + ".xml", "w")
    annotFile.write(annotation)

    multIm2.save( imgSaveDir + "/" + multImgFileBasename + ".jpg" )
    multIm2.close()
    multIm3.save( gtImgSaveDir + "/" + multImgFileBasename + ".jpg" )
    multIm3.close()
    
    #print imgFileBasename + ".jpg & .xml saved"
                    
trainlistFilename = imgSetsSaveDir + "/trainmult.txt"

if appendMode:
    openMode = "a"
else:
    openMode = "w"
    
FTrainList = open(trainlistFilename, openMode)
for imgFilename in imglist:
	  FTrainList.write(imgFilename + "\n")
FTrainList.close()

'''
imglist_npobj = np.zeros( (len(imglist),), dtype=np.object)
imglist_npobj[:] = imglist
imglist_npobj = imglist_npobj.reshape(-1,1)
gtBoxes_npobj = np.zeros( (len(imglist),), dtype=np.object)
gtBoxes_npobj[:] = img_gtBoxes
gtBoxes_npobj = gtBoxes_npobj.reshape(-1,1)

matFilename = proposalDir + "/bread_trainmult.mat"
if appendMode and isfile(matFilename):
    oldmat = sio.loadmat(matFilename)
    gtBoxes_npobj0 = oldmat['boxes']
    imglist_npobj0 = oldmat['images']
    gtBoxes_npobj1 = np.concatenate( (gtBoxes_npobj0, gtBoxes_npobj) )
    imglist_npobj1 = np.concatenate( (imglist_npobj0, imglist_npobj) )
else:
    gtBoxes_npobj1 = gtBoxes_npobj
    imglist_npobj1 = imglist_npobj
    
sio.savemat( matFilename, { 'boxes': gtBoxes_npobj1, 'images': imglist_npobj1 } )
'''
