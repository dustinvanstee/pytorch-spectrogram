Directory structure should look like this ...
01_rawVideos  02_splitVideos  03_labeledSplitVideos  04_labeledSplitPngs  05_train  05_val

01_rawVideos - contains the golden source videos with both vides and audio channel.  No sub-directories
02_splitVideos - contains split up video/audio clips in chunks (1000ms default typical).  This folder is manually populated
               - subdirectories [1000ms, 500ms, etc]
03_labeledSplitVideos - contains same structure as 02_splitVideos with added subdirectory.  The second sub-directory specifies class name.  This folder is manually populated.
              - subdirectories [1000ms, 500ms, etc]
              - sub-subdirectories [Luke, Darth , Background, etc]

04_labeledSplitPngs - This folder contains a 1 to 1 correspondence with 03_labeledSplitVideos and is automatically generated.
05_train, 05_val   -  This folder is just a subset of 04_labeledSplitPngs based on random split and is automatically generated.

