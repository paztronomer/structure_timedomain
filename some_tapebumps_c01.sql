select im.nite, im.expnum, im.band, im.exptime, 
    '/archive_data/desarchive/'||fai.path||'/'||fai.filename||'.fz' 
	from proctag pt, file_archive_info fai, image im 
	where pt.tag='Y4A1_FINALCUT' 
and pt.pfw_attempt_id=im.pfw_attempt_id 
and im.filename=fai.filename 
and im.nite between 20161013 and 20161101 
and fai.filename like '%_c01_%' 
and im.filetype='red_immask' 
order by fai.filename;
