def draw_bounding_box(img, left, top, right, bottom, class_color, label, confidence):
  overlay = img.copy()
  cv2.rectangle(img, (left, top), (right, bottom), color=class_color, thickness=-1)
  alpha = .8
  img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
  cv2.rectangle(img, (left, top), (right, bottom), color=class_color, thickness=1)
  cv2.putText(img, "{} - {:.1f}".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, class_color, thickness=1)
  #cv2_imshow(image_new)
  return img


def getPrediction(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio