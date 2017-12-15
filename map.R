library(leaflet)
library(dplyr)

IL_LOC = read.csv("~/Yiming/IL_loc.csv", header = TRUE)
IL_RENTER_LRF = read.csv("~/Yiming/IL_RENTER_LRF.csv", header = TRUE)
IL_HO_LRF = read.csv("~/Yiming/IL_HO_LRF.csv", header = TRUE)
IL_HO_PREM = read.csv("~/Yiming/IL_HO_PREM.csv", header = TRUE)
try_2017 = read.csv("~/Yiming/try_2017.csv", header = TRUE)
try_2017 = try_2017[try_2017$policy_cnt_inforce_true >= 1, ] %>%
  filter(ZONE %in% c(30, 31, 40, 50))

# merge two datasets
IL_RENTER_LOC_LRF = left_join(IL_LOC, IL_RENTER_LRF[, -c(5, 7)], by = c("GRID_ID" = "GRID_ID")) %>% na.omit()


# Visualize LRFs
# Create a continuous palette function
pal <- colorNumeric(
  palette = "Reds",
  domain = IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF <= 1.41, ]$CALC_LRF)


leaflet(IL_RENTER_LOC_LRF) %>% addTiles() %>%
  addCircles(lng = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF <= 1.4 &IL_RENTER_LOC_LRF$CALC_LRF >= 0.9 , ]$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF <= 1.4 &IL_RENTER_LOC_LRF$CALC_LRF >= 0.9 , ]$LATUD_NUM, weight = 1,  color = ~palho(IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF <= 1.4 &IL_RENTER_LOC_LRF$CALC_LRF >=0.9 , ]$CALC_LRF),
             radius = 500, fillOpacity = .8, group = "CALC LRF <= 1.4") %>%
  addLegend("bottomright", pal = palho, values = IL_HO_LOC_LRF$BAL_IND_FCTR, title = "Renter CALC LRF", opacity = 1) %>%
  
  addCircles(lng = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF < 0.9 , ]$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF < 0.9 , ]$LATUD_NUM, weight = 1,  color = palho(0.9),
             radius = 500, fillOpacity = .8, group = "CALC LRF > 1.4") %>%
  
  addCircles(lng = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF > 1.4, ]$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$CALC_LRF > 1.4, ]$LATUD_NUM, weight = 1,  color = palho(1.4),
             radius = 500, fillOpacity = .8, group = "CALC LRF > 1.4") %>%
  addLegend("bottomright", colors = c(palho(1.4)), labels = c("CALC LRF > 1.4"), opacity = 1) %>%
  
  addLayersControl(
    overlayGroups = c("CALC LRF <= 1.4", "CALC LRF > 1.4"),
    options = layersControlOptions(collapsed = FALSE)) %>% hideGroup("CALC LRF > 1.4")


# Visualize Zone factors
IL_RENTER_LRF$ZONE_FCTR = ifelse(IL_RENTER_LRF$ZONE == 30, 1.064, 
                                 ifelse(IL_RENTER_LRF$ZONE == 31, 1.389, 
                                        ifelse(IL_RENTER_LRF$ZONE == 40, 0.739, 0.915)))

IL_RENTER_LOC_LRF_ZONE = left_join(IL_LOC, IL_RENTER_LRF[, c(1, 2, 7)], by = c("GRID_ID" = "GRID_ID")) %>% 
  filter(ZONE == 30 | ZONE == 31 | ZONE == 40 | ZONE == 50) %>%
  na.omit()

factpal = colorFactor(topo.colors(5),IL_RENTER_LOC_LRF_ZONE$ZONE)

leaflet(IL_RENTER_LOC_LRF_ZONE) %>% addTiles() %>%
  addCircles(lng = ~IL_RENTER_LOC_LRF_ZONE$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF_ZONE$LATUD_NUM, weight = 1,  color = ~factpal(ZONE),
             radius = 200, fillOpacity = .8) %>%
  addLegend("bottomright", pal = factpal, values = IL_RENTER_LOC_LRF_ZONE$ZONE, title = "Zone", opacity = 1) %>%
  
  addPopups(-88.5, 41, "Zone Factor = 0.915",
            options = popupOptions(closeButton = TRUE)
  ) %>%
  
  addPopups(-88.2, 41.85, "Zone Factor = 0.739",
            options = popupOptions(closeButton = TRUE)
  ) %>%
  
  addPopups(-87.63, 41.68, "Zone Factor = 1.389",
            options = popupOptions(closeButton = TRUE)
  ) %>%
  
  addPopups(-87.78, 41.95, "Zone Factor = 1.064",
            options = popupOptions(closeButton = TRUE)
  ) 




# Visualiza Loss Ratio
IL_RENTER_LOC_LR = left_join(IL_LOC, IL_RENTER_LRF, by = c("GRID_ID" = "GRID_ID"))

temp1 = IL_RENTER_LOC_LR[-which(is.na(IL_RENTER_LOC_LR$LR)),] # exclude LR == NA

pal <- colorNumeric(
  palette = "Reds",
  domain = seq(0, 3, 0.5))


leaflet(temp1) %>% addTiles() %>%
  addCircles(lng = ~temp1$LNGTD_NUM, lat = ~temp1$LATUD_NUM, weight = 1,  color = ~pal(temp1$LR),
             radius = 500, fillOpacity = .8) %>%
  addLegend("bottomright", pal = pal, values = seq(0, 3, 0.5), title = "Renter Loss Ratio", opacity = 1)

 
# Visualiza EARNED PREMIUM
pal <- colorNumeric(
  palette = "Reds",
  domain = IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM <= 5000, ]$EARNED_PREM)


leaflet(IL_RENTER_LOC_LRF) %>% addTiles() %>%
  addCircles(lng = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM <= 5000, ]$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM <= 5000, ]$LATUD_NUM, weight = 1,  color = ~pal(IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM <= 5000, ]$EARNED_PREM),
             radius = 50, fillOpacity = .8, group = "EARNED PREMIUM <= 5000") %>%
  addLegend("bottomright", pal = pal, values = seq(0, 5000, 1000), title = "Renter Earned Premium <= 5000", opacity = 1) %>%
  
  addCircles(lng = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM > 5000, ]$LNGTD_NUM, lat = ~IL_RENTER_LOC_LRF[IL_RENTER_LOC_LRF$EARNED_PREM > 5000, ]$LATUD_NUM, weight = 1,  color = "blue",
             radius = 100, fillOpacity = .8, group = "EARNED PREMIUM > 5000") %>%
  addLegend("bottomright", colors = c("blue"), labels = c("Renter Earned Premium > 5000"), opacity = 1) %>%
  
  addLayersControl(
    overlayGroups = c("EARNED PREMIUM <= 5000", "EARNED PREMIUM > 5000"),
    options = layersControlOptions(collapsed = FALSE)) %>% hideGroup("EARNED PREMIUM > 5000")

##################### For Homeowners ##########################
# merge two datasets
IL_HO_LOC_LRF = left_join(IL_LOC, IL_HO_LRF, by = c("GRID_ID" = "GRID_ID")) %>% na.omit()
IL_HO_LOC_LRF = left_join(IL_HO_LOC_LRF, IL_HO_PREM, by = c("GRID_ID" = "GRID_ID")) %>% na.omit()

# Visualize LRFs
# Create a continuous palette function
palho <- colorNumeric(
  palette = "Reds",
  domain = IL_HO_LOC_LRF$BAL_IND_FCTR)


leaflet(IL_HO_LOC_LRF) %>% addTiles() %>%
  addCircles(lng = ~IL_HO_LOC_LRF$LNGTD_NUM, lat = ~IL_HO_LOC_LRF$LATUD_NUM, weight = 1,  color = ~palho(IL_HO_LOC_LRF$BAL_IND_FCTR),
             radius = 10, fillOpacity = .8) %>%
  addLegend("bottomright", pal = palho, values =IL_HO_LOC_LRF$BAL_IND_FCTR, title = "Homeowners' Balanced LRFs", opacity = 1)


# Visualiza EARNED PREMIUM
pal <- colorNumeric(
  palette = "Reds",
  domain = IL_HO_LOC_LRF$EARNED_PREM)


leaflet(IL_HO_LOC_LRF) %>% addTiles() %>%
  addCircles(lng = ~IL_HO_LOC_LRF$LNGTD_NUM, lat = ~IL_HO_LOC_LRF$LATUD_NUM, weight = 1,  color = ~pal(IL_HO_LOC_LRF$EARNED_PREM),
             radius = 50, fillOpacity = .8) %>%
  addLegend("bottomright", pal = pal, values = IL_HO_LOC_LRF$EARNED_PREM, title = "Homeowners Earned Premium", opacity = 1)


#################### Homeowners vs Renters ####################


IL_HO_RENTER_LRF = left_join(IL_HO_LOC_LRF, IL_RENTER_LOC_LRF[, c("GRID_ID", "CALC_LRF")], by = c("GRID_ID" = "GRID_ID")) %>% na.omit()

library(ggplot2)
ggplot(IL_HO_RENTER_LRF) + 
  geom_density(aes(BAL_IND_FCTR, color = "BAL_IND_FCTR")) + 
  geom_density(aes(CALC_LRF, color = "CALC_LRF")) +
  labs(title = "LRF Type", x = "TY [°C]", y = "Txxx", color = "Legend Title\n") 

IL_HO_RENTER_LRF$DIFF = IL_HO_RENTER_LRF$BAL_IND_FCTR - IL_HO_RENTER_LRF$CALC_LRF
#IL_HO_RENTER_LRF$CALC_LRF_SCALE = (IL_HO_RENTER_LRF$CALC_LRF - mean(IL_HO_RENTER_LRF$BAL_IND_FCTR)) / sd(IL_HO_RENTER_LRF$BAL_IND_FCTR)


pal <- colorNumeric(
  palette = "RdBu",
  domain = seq(-0.5, 0.5, 0.2))


leaflet(IL_HO_RENTER_LRF) %>% addTiles() %>%
  addCircles(lng = ~IL_HO_RENTER_LRF[IL_HO_RENTER_LRF$DIFF >= -0.5 & IL_HO_RENTER_LRF$DIFF <= 0.5, ]$LNGTD_NUM, lat = ~IL_HO_RENTER_LRF[IL_HO_RENTER_LRF$DIFF >= -0.5 & IL_HO_RENTER_LRF$DIFF <= 0.5, ]$LATUD_NUM, weight = 1,  color = ~pal(IL_HO_RENTER_LRF[IL_HO_RENTER_LRF$DIFF >= -0.5 & IL_HO_RENTER_LRF$DIFF <= 0.5, ]$DIFF),
             radius = 1000, fillOpacity = .8) %>%
  
  addCircles(lng = ~IL_HO_RENTER_LRF[IL_HO_RENTER_LRF$DIFF < -0.5, ]$LNGTD_NUM, lat = ~IL_HO_RENTER_LRF[IL_HO_RENTER_LRF$DIFF < -0.5, ]$LATUD_NUM, weight = 1,  color = pal(-0.5),
             radius = 1000, fillOpacity = .8) %>%
  
  addLegend("bottomright", pal = pal, values = seq(-0.5, 0.5, 0.2), title = "HO LRF - RENTER LRF", opacity = 1)




######################### Test predicted loss ratios for 2017 IL renters #####################################
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

############# Loss Ratio (ZSZ) ##############
# Zone 40: 0.739, Zone 50: 0.915, Zone 30: 1.064, Zone 31: 1.389
# get rid of all outliers
x1 = try_2017[try_2017$ZONE == 40, ]$LOSS_RATIO_OLD
x1 = x1[!x1 %in% boxplot.stats(x1)$out]

x2 = try_2017[try_2017$ZONE == 50, ]$LOSS_RATIO_OLD
x2 = x2[!x2 %in% boxplot.stats(x2)$out]

x3 = try_2017[try_2017$ZONE == 30, ]$LOSS_RATIO_OLD
x3 = x3[!x3 %in% boxplot.stats(x3)$out]

x4 = try_2017[try_2017$ZONE == 31, ]$LOSS_RATIO_OLD
x4 = x4[!x4 %in% boxplot.stats(x4)$out]

p1 = qplot(x1, geom="histogram", fill = I("orange"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 40 (0.739)", xlab = "Current loss ratio", bins = 50)

p2 = qplot(x2, geom="histogram", fill = I("orange"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 50 (0.915)", xlab = "Current loss ratio", bins = 50)

p3 = qplot(x3, geom="histogram", fill = I("orange"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 30 (1.064)", xlab = "Current loss ratio", bins = 50)

p4 = qplot(x4, geom="histogram", fill = I("orange"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 31 (1.389)", xlab = "Current loss ratio", bins = 50)

multiplot(p1, p2, p3, p4, cols = 4)

# work for Monday: boxblots of each zone 

############# Loss Ratio (Using non-tenant LRF, arithmetic weighting) ##############
# Zone 40: 0.739, Zone 50: 0.915, Zone 30: 1.064, Zone 31: 1.389
# get rid of all outliers
x5 = try_2017[try_2017$ZONE == 40, ]$LOSS_RATIO_NEW
x5 = x5[!x5 %in% boxplot.stats(x5)$out]

x6 = try_2017[try_2017$ZONE == 50, ]$LOSS_RATIO_NEW
x6 = x6[!x6 %in% boxplot.stats(x6)$out]

x7 = try_2017[try_2017$ZONE == 30, ]$LOSS_RATIO_NEW
x7 = x7[!x7 %in% boxplot.stats(x7)$out]

x8 = try_2017[try_2017$ZONE == 31, ]$LOSS_RATIO_NEW
x8 = x8[!x8 %in% boxplot.stats(x8)$out]


p5 = qplot(x5, geom="histogram", fill = I("green"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 40 (0.739)", xlab = "Loss ratio by LRF", bins = 50)

p6 = qplot(x6, geom="histogram", fill = I("green"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 50 (0.915)", xlab = "Loss ratio by LRF", bins = 50)

p7 = qplot(x7, geom="histogram", fill = I("green"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 30 (1.064)", xlab = "Loss ratio by LRF", bins = 50)

p8 = qplot(x8, geom="histogram", fill = I("green"), xlim = c(0, 15), ylim = c(0, 150), col = I("red"), alpha=I(.2), main = "Zone 31 (1.389)", xlab = "Loss ratio by LRF", bins = 50)

multiplot(p1, p5, p2, p6, p3, p7, p4, p8, cols = 4)



######################### Simple Statistics ###########################
try_2017[is.na(try_2017)] = 0

# Zone/ Subzone
old_zone_prem = try_2017 %>%
  group_by(ZONE) %>%
  summarise(old_zone_prem = sum(earned_prem))

old_zone_loss = try_2017 %>%
  group_by(ZONE) %>%
  summarise(old_zone_loss = sum(case_incurred))

old_zone_lr = left_join(old_zone_prem, old_zone_loss, by = c("ZONE" = "ZONE")) %>%
  group_by(ZONE) %>%
  summarise(old_zone_lr = old_zone_loss/old_zone_prem)

# variances within each zone
try_2017 %>%
  group_by(ZONE) %>%
  summarise(xx = var(LOSS_RATIO_OLD))

statewide_old = sum(try_2017$case_incurred) / sum(try_2017$earned_prem)

old_zone_lr$old_zone_lr = old_zone_lr$old_zone_lr / statewide_old





# LRF method
new_zone_prem = try_2017 %>%
  group_by(ZONE) %>%
  summarise(new_zone_prem = sum(NEW_PREM))

new_zone_loss = try_2017 %>%
  group_by(ZONE) %>%
  summarise(new_zone_loss = sum(case_incurred))

new_zone_lr = left_join(new_zone_prem, new_zone_loss, by = c("ZONE" = "ZONE")) %>%
  group_by(ZONE) %>%
  summarise(new_zone_lr = new_zone_loss/new_zone_prem)

# variances within each zone
try_2017 %>%
  group_by(ZONE) %>%
  summarise(xx = var(LOSS_RATIO_NEW))

statewide_new = sum(try_2017$case_incurred) / sum(try_2017$NEW_PREM)

new_zone_lr$new_zone_lr = new_zone_lr$new_zone_lr / statewide_new





try_2017 %>%
  summarise(mean_lr = mean(LOSS_RATIO_NEW))

mean(old_zone_lr$old_zone_lr) # 2.446

# LRF means
mean(new_zone_lr$new_zone_lr) # 2.277

# Zone/Subzone variances
var(try_2017$LOSS_RATIO_OLD) # 7020.838

# LRF variances
var(try_2017$LOSS_RATIO_NEW) # 4891.051
