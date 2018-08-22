--SET ECHO OFF NEWP 0 SPA 1 PAGES 0 FEED OFF HEAD OFF TRIMS ON LINESIZE 1000

--spool dboper_y4e1_a01.lst

--select im.nite, im.expnum, im.band, im.exptime, im.ccdnum, fai.path 
select im.expnum, fai.path 
	from proctag pt, file_archive_info fai, image im 
	where pt.tag='Y4A1_FINALCUT' 
and pt.pfw_attempt_id=im.pfw_attempt_id 
and im.filename=fai.filename 
and im.nite between 20161018 and 20161208 --and fai.filename like '%_c01_%' 
and im.filetype='red_immask';

--exit

--20170103 and 20170218
--20161209 and 20170102 
--20160813 and 20161208 --20161013 and 20161101 
