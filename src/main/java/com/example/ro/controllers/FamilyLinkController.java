package com.example.ro.controllers;

import com.example.ro.dto.requestDTO.LinkDTO;
import com.example.ro.dto.requestDTO.UpdateLinkDTO;
import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.services.FamilyLinkService;
import jakarta.validation.Valid;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/family-link")
@AllArgsConstructor
public class FamilyLinkController {

    private final FamilyLinkService familyLinkService;

    @PostMapping
    private ApiError createLink ( @Valid @RequestBody LinkDTO dto) {
        return familyLinkService.createLink(dto);
    }

    @GetMapping("/dijkstra")
    public ApiError getShortestPathByDijkstra(@RequestParam int startId, @RequestParam int endId, @RequestParam int familyTreeId) {
        return familyLinkService.findShortestPathByDijkstra(familyTreeId, startId, endId);
    }

    @PatchMapping("/{id}")
    public ApiError updateLink(@PathVariable int id, @RequestBody UpdateLinkDTO updateLinkDTO){
        return familyLinkService.updadeLink(id, updateLinkDTO);
    }

    @DeleteMapping("/{id}")
    public ApiError deleteLink(@PathVariable int id) {
        return familyLinkService.deleteLink(id);
    }


    //    @GetMapping("/bellman_ford")
//    public ApiError getShortestPathByBellmanFord(@RequestParam int startId, @RequestParam int endId, @RequestParam int familyTreeId) {
//        return familyLinkService.findShortestPathByBellmanFord(familyTreeId, startId, endId);
//    }
}
